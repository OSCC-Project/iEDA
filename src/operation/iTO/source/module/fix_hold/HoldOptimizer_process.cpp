// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "../../config/ToConfig.h"
#include "EstimateParasitics.h"
#include "HoldOptimizer.h"
#include "Master.h"
#include "Placer.h"
#include "Point.h"
#include "Reporter.h"
#include "data_manager.h"
#include "timing_engine.h"

using namespace std;

namespace ito {

void HoldOptimizer::process()
{
  TOSlack worst_timing_slack_hold = timingEngine->getWorstSlack(AnalysisMode::kMin);
  int iteration = 1;
  int number_insert_buffer = 1;
  while (worst_timing_slack_hold < _target_slack) {
    number_insert_buffer = checkAndOptimizeHold();
    worst_timing_slack_hold = timingEngine->getWorstSlack(AnalysisMode::kMin);
    toRptInst->get_ofstream() << "\nThe " << iteration << "-th optimization insert " << number_insert_buffer
                              << " buffer. \nCurrent worst hold slack is " << worst_timing_slack_hold
                              << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
    if (number_insert_buffer > 0) {
      toEvalInst->estimateAllNetParasitics();
      timingEngine->get_sta_engine()->updateTiming();
    } else {
      // break optimization iteration if no buffer inserted.
      break;
    }
    iteration++;
  }
}

/**
 * @brief step .1: check and find all hold violation endpoins;
 * step .2: optimize hold violation path.
 * 
 * @return int, the number of insert buffer
 */
int HoldOptimizer::checkAndOptimizeHold()
{
  int number_insert_buffer = 0;

  bool exit_vioaltion = checkAndFindVioaltion();

  if (exit_vioaltion) {
    number_insert_buffer = performOptimizationProcess();
  } else {
    toRptInst->get_ofstream() << "TO: There are no hold violations found in current design.\n";
  }

  toRptInst->get_ofstream() << "TO: insert " << number_insert_buffer << " hold buffers when fix hold.\n";
  toRptInst->get_ofstream().close();
  return number_insert_buffer;
}

/**
 * @brief step .1: check and find all hold violation endpoins;
 * 
 * @return true, find violation
 * @return false, no violation
 */
bool HoldOptimizer::checkAndFindVioaltion()
{
  TOSlack worst_slack;
  bool exit_vioaltion = isFindEndpointsWithHoldViolation(worst_slack);
  return exit_vioaltion;
}

bool HoldOptimizer::isFindEndpointsWithHoldViolation(TOSlack& worst_slack)
{
  worst_slack = kInf;
  _end_pts_hold_violation.clear();
  bool is_find = false;

  for (auto* end : _all_end_points) {
    TOSlack slack = timingEngine->getWorstSlack(end, AnalysisMode::kMin);
    worst_slack = min(worst_slack, slack);
    if (slack < _target_slack) {
      _end_pts_hold_violation.insert(end);
      is_find = true;
    }
  }
  return is_find;
}

/**
 * @brief step .2: optimize hold violation path. Include optimization process.
 * 
 * @return int, the number of insert buffer 
 */
int HoldOptimizer::performOptimizationProcess()
{
  int number_insert_buffer = 0;
  toRptInst->get_ofstream() << "\nTO: >>>>>>>>> Beign hold optimization! \n\t\t\tHold target slack -> " << _target_slack << endl
                            << "\nTO: Total find " << _end_pts_hold_violation.size()
                            << " endpoints with hold violations in current design.\n";
  toRptInst->get_ofstream().close();

  if (toDmInst->get_buffer_num() > _max_numb_insert_buf || toDmInst->reachMaxArea()) {
    return number_insert_buffer;
  }

  bool exit_vioaltion = true;
  while (exit_vioaltion) {
    int old = toDmInst->get_buffer_num();
    optimizeHoldViolation();
    int number_insert_buf_this_opt = toDmInst->get_buffer_num() - old;

    number_insert_buffer += number_insert_buf_this_opt;
    // Ensure effective optimization and avoid dead ends
    if (number_insert_buf_this_opt == 0) {
      break;
    }

    toEvalInst->excuteParasiticsEstimate();
    timingEngine->get_sta_engine()->updateTiming();

    exit_vioaltion = checkAndFindVioaltion();

    if (toDmInst->get_buffer_num() > _max_numb_insert_buf) {
      toRptInst->report("TO: Reach the maximum number of buffers that can be inserted.\n");
      break;
    }
    if (toDmInst->reachMaxArea()) {
      toRptInst->report("TO: Reach the maximum utilization of current design.\n");
      break;
    }
  }
  return number_insert_buffer;
}

void HoldOptimizer::optimizeHoldViolation() {
  TOVertexSeq fanins = getFanins(_end_pts_hold_violation);
  optHoldViolationEnd(fanins);
}

bool HoldOptimizer::findEndpointsWithHoldViolation(TOVertexSet end_points, TOSlack& worst_slack, TOVertexSet& hold_violations)
{
  worst_slack = kInf;
  hold_violations.clear();
  bool is_find = false;

  for (auto* end : end_points) {
    TOSlack slack = timingEngine->getWorstSlack(end, AnalysisMode::kMin);
    worst_slack = min(worst_slack, slack);
    if (slack < _target_slack) {
      hold_violations.insert(end);
      is_find = true;
    }
  }
  return is_find;
}

int HoldOptimizer::optHoldViolationEnd(TOVertexSeq fanins)
{
  int insert_buffer_count = 0;

  int batch_size = fanins.size() / 10;
  for (int i = 0; i < fanins.size(); i++) {
    if ((i + 1) % batch_size == 0 || (i + 1) == fanins.size()) {
      LOG_INFO << "Opt hold end: " << (i + 1) << "/" << fanins.size() << "(" << (double(i + 1) / fanins.size()) * 100 << "%)\n";
    }

    auto vertex = fanins[i];
    TODesignObjSeq load_vio_pins;

    TODelay max_insert_delay = std::numeric_limits<TODelay>::max();

    TODesignObjSeq loads = vertex->get_design_obj()->get_net()->getLoads();
    for (auto load_obj : loads) {
      StaVertex* fanout_vertex = timingEngine->get_sta_engine()->findVertex(load_obj->getFullName().c_str());

      if (!fanout_vertex) {
        continue;
      }

      TOSlack timing_slack_hold = timingEngine->getWorstSlack(fanout_vertex, AnalysisMode::kMin);
      TOSlack timing_slacks_setup = timingEngine->getWorstSlack(fanout_vertex, AnalysisMode::kMax);
      if (timing_slack_hold < _target_slack) {
        TODelay delay
            = _allow_setup_violation ? _target_slack - timing_slack_hold : min(_target_slack - timing_slack_hold, timing_slacks_setup);

        if (delay > 0.0) {
          max_insert_delay = min(max_insert_delay, delay);
          // add load with hold violation.
          load_vio_pins.push_back(fanout_vertex->get_design_obj());
        }
      }
    }  // for all loads

    if (load_vio_pins.empty()) {
      return insert_buffer_count;
    }

    int insert_number = std::ceil(max_insert_delay / _hold_insert_buf_cell_delay);

    insert_buffer_count += insert_number;
    insertBufferOptHold(vertex, insert_number, load_vio_pins);

    if (toDmInst->get_buffer_num() > _max_numb_insert_buf || toDmInst->reachMaxArea()) {
      LOG_INFO << "Reach the maximum number of buffers that can be inserted.\n";
      return insert_buffer_count;
    }
  }
  return insert_buffer_count;
}

void HoldOptimizer::reportWNSAndTNS()
{
  toRptInst->get_ofstream() << "\n---------------------------------------------------------------------------\n"
                            << setiosflags(ios::left) << setw(35) << "Clock Group" << resetiosflags(ios::left) << setiosflags(ios::right)
                            << setw(20) << "Hold TNS" << setw(20) << "Hold WNS" << resetiosflags(ios::right) << endl
                            << "---------------------------------------------------------------------------" << endl;
  auto clk_list = timingEngine->get_sta_engine()->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto tns1 = timingEngine->get_sta_engine()->getTNS(clk_name, AnalysisMode::kMin);
    auto wns1 = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMin);
    toRptInst->get_ofstream() << setiosflags(ios::left) << setw(35) << clk_name << resetiosflags(ios::left) << setiosflags(ios::right)
                              << setw(20) << tns1 << setw(20) << wns1 << resetiosflags(ios::right) << endl;
  }
  toRptInst->get_ofstream() << "---------------------------------------------------------------------------" << endl;
  toRptInst->get_ofstream().close();
}

/**
 * @brief calc vertex max/min, rise/fall slack
 *
 * @param vertex
 * @param slacks
 */
void HoldOptimizer::calcStaVertexSlacks(StaVertex* vertex, TOSlacks slacks)
{
  vector<AnalysisMode> analy_mode = {AnalysisMode::kMax, AnalysisMode::kMin};
  vector<TransType> rise_fall = {TransType::kRise, TransType::kFall};

  for (auto mode : analy_mode) {
    for (auto rf : rise_fall) {
      auto pin_slack = vertex->getSlackNs(mode, rf);
      TOSlack slack = pin_slack ? *pin_slack : kInf;
      int mode_idx = (int) mode - 1;
      int rf_idx = (int) rf - 1;
      slacks[rf_idx][mode_idx] = slack;
    }
  }
}

TOSlack HoldOptimizer::calcSlackGap(StaVertex* vertex)
{
  TOSlacks slacks;
  calcStaVertexSlacks(vertex, slacks);

  return min(slacks[TYPE_RISE][_mode_max] - slacks[TYPE_RISE][_mode_min], slacks[TYPE_FALL][_mode_max] - slacks[TYPE_FALL][_mode_min]);
}

TOVertexSet HoldOptimizer::getEndPoints()
{
  TOVertexSet end_points;
  auto* ista = timingEngine->get_sta_engine()->get_ista();
  StaGraph* the_graph = &(ista->get_graph());
  StaVertex* vertex;
  FOREACH_END_VERTEX(the_graph, vertex)
  {
    end_points.insert(vertex);
  }
  return end_points;
}

TOVertexSeq HoldOptimizer::getFanins(TOVertexSet end_points)
{
  TOVertexSet fanins;
  fanins.clear();
  for (auto* end_point : end_points) {
    auto net = end_point->get_design_obj()->get_net();
    auto driver = net->getDriver();
    auto vertex = timingEngine->get_sta_engine()->findVertex(driver->getFullName().c_str());
    fanins.insert(vertex);
  }
  auto fanins_seq = TOVertexSeq(fanins.begin(), fanins.end());
  sort(fanins_seq.begin(), fanins_seq.end(), [=, this](StaVertex* v1, StaVertex* v2) {
    auto v1_slack = v1->getSlack(AnalysisMode::kMin, TransType::kRise);
    auto v2_slack = v2->getSlack(AnalysisMode::kMin, TransType::kRise);
    TOSlack s1 = v1_slack ? *v1_slack : kInf;
    TOSlack s2 = v2_slack ? *v2_slack : kInf;
    return s1 < s2;
  });
  return fanins_seq;
};

}  // namespace ito
