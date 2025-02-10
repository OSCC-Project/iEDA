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
#include "EstimateParasitics.h"
#include "Placer.h"
#include "Reporter.h"
#include "SetupOptimizer.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "data_manager.h"
#include "liberty/Lib.hh"
#include "timing_engine.h"

using namespace std;

namespace ito {

/**
 * @brief optimize the setup violation, including optimization process and report.
 * 
 * @param end_pts_setup_violation 
 */
void SetupOptimizer::optimizeViolationProcess(TOVertexSeq& end_pts_setup_violation)
{
  StaSeqPathData* worst_path = worstRequiredPath();
  TOSlack worst_slack = worst_path->getSlackNs();
  toRptInst->get_ofstream() << "\n-----------------------------------------\n"
                            << setiosflags(ios::right) << setw(20) << "Setup TNS" << setw(20) << "Setup WNS" << resetiosflags(ios::right)
                            << endl
                            << "-----------------------------------------" << endl;
  reportWNSAndTNS();

  int number_of_decreasing_slack_iter = 0;
  int iter = 0;
  int max_iter = max((int) (end_pts_setup_violation.size() * toConfig->get_optimize_endpoints_percent()), 10);
  for (auto node : end_pts_setup_violation) {
    iter++;
    cout << ">>>>>>>>>>>>>>>>>>>>>>>> optimize the " << iter << "-th endpoints, max optimized number is "
         << max_iter << "." << endl;
    if (iter > max_iter) {
      break;
    }

    TOSlack prev_worst_slack = -kInf;
    while (worst_slack < toConfig->get_setup_target_slack()) {
      optimizeSetupViolation(node, true, false);

      incrUpdateRCAndTiming();

      auto worst_slack_exist = timingEngine->getNodeWorstSlack(node);
      if (worst_slack_exist == std::nullopt) {
        break;
      }
      worst_slack = *worst_slack_exist;

      if (checkSlackDecrease(worst_slack, prev_worst_slack, number_of_decreasing_slack_iter)) {
        break;
      }
      prev_worst_slack = worst_slack;
    }
  }

  toEvalInst->estimateAllNetParasitics();
  timingEngine->get_sta_engine()->updateTiming();
}

void SetupOptimizer::incrUpdateRCAndTiming() {
  auto nets_for_update = toEvalInst->get_parasitics_invalid_net();
  for (auto net_up : nets_for_update) {
    auto net_pins = net_up->get_pin_ports();
    for (auto pin_port : net_pins) {
      if (pin_port->isPort()) {
        continue;
      }
      auto inst_name = pin_port->get_own_instance()->getFullName();
      timingEngine->get_sta_engine()->moveInstance(inst_name.c_str(), 20);
    }
  }
  toEvalInst->excuteParasiticsEstimate();
  timingEngine->get_sta_engine()->incrUpdateTiming();
}

bool SetupOptimizer::checkSlackDecrease(TOSlack& current_slack, TOSlack& last_slack, int& number_of_decreasing_slack_iter)
{
  if (approximatelyLessEqual(current_slack, last_slack)) {
    float diff = last_slack - current_slack;
    if (diff > 0.02 * abs(last_slack)) {
      return true;
    }

    number_of_decreasing_slack_iter++;
    if (number_of_decreasing_slack_iter > toConfig->get_number_of_decreasing_slack_iter()) {
      return true;
    }
  } else {
    number_of_decreasing_slack_iter = 0;
  }

  auto wns = timingEngine->getWNS();
  if (wns > toConfig->get_setup_target_slack()) {
    return true;
  }

  return false;
}

/**
 * @brief net-based optimization. 
 * TODO: vertex can induce timing path, thus perform path-based optimization
 * 
 * @param vertex violation endpoint
 * @param perform_gs whether perform gate sizing
 * @param perform_buf whether perform buffer insertion
 */
void SetupOptimizer::optimizeSetupViolation(StaVertex *vertex, bool perform_gs, bool perform_buf) {
  auto shouldGateSize = [&](vector<TimingEngine::PathNet> init,
                            TimingEngine::PathNet path_net, int &idx) {
    vector<TimingEngine::PathNet>::iterator itr =
        find(init.begin(), init.end(), path_net);
    idx = distance(init.begin(), itr);
    if (idx >= 1) {
      return true;
    }
    return false;
  };

  auto getInputPin = [&](const vector<TimingEngine::PathNet> &path_driver_vertexs,
                         const TimingEngine::PathNet         &path) {
    auto       in_path = path_driver_vertexs[distance(path_driver_vertexs.begin(),
                                                      find(path_driver_vertexs.begin(),
                                                           path_driver_vertexs.end(), path)) -
                                       1];
    StaVertex *in_vertex = in_path.load;
    StaVertex *driver_vertex = in_path.driver;
    auto      *in_obj = in_vertex->get_design_obj();
    auto      *driver_obj = driver_vertex->get_design_obj();
    return make_pair(dynamic_cast<Pin *>(in_obj), dynamic_cast<Pin *>(driver_obj));
  };

  auto                          worst_path = timingEngine->getNodeWorstPath(vertex);
  vector<TimingEngine::PathNet> path_driver_vertexs =
      timingEngine->get_sta_engine()->getPathDriverVertexs(worst_path);
  int path_length = path_driver_vertexs.size();

  if (path_length <= 1) {
    return;
  }

  vector<TimingEngine::PathNet> sorted_path_driver_vertexs;
  for (int i = 0; i < path_length; i++) {
    auto path = path_driver_vertexs[i];
    sorted_path_driver_vertexs.push_back(path);
  }
  sort(sorted_path_driver_vertexs.begin(), sorted_path_driver_vertexs.end(),
       [](TimingEngine::PathNet n1, TimingEngine::PathNet n2) {
         return n1.delay > n2.delay;
       });

  for (int i = 0; i < (int)path_length; i++) {
    auto       path = sorted_path_driver_vertexs[i];
    StaVertex *driver_vertex = path.driver;
    auto      *obj = driver_vertex->get_design_obj();
    Pin       *driver_pin = dynamic_cast<Pin *>(obj);

    float cap_load = driver_pin->get_net()->getLoad(AnalysisMode::kMax, TransType::kRise);

    if (perform_gs) {
      int driver_idx = 0;
      if (shouldGateSize(path_driver_vertexs, path, driver_idx)) {
        auto [in_pin, driver_pin] = getInputPin(path_driver_vertexs, path);
        float driver_res = driver_pin->get_cell_port()->driveResistance();
        if (in_pin && performGateSizing(cap_load, driver_res, in_pin, driver_pin)) {
          break;
        }
      }
    }

    int fanout = getFanoutNumber(driver_pin);
    if (perform_buf) {
      if (performBufferingIfNecessary(driver_pin, fanout)) {
        break;
      }
    }

    if (performSplitBufferingIfNecessary(driver_vertex, fanout)) {
      break;
    }
  }
}

}  // namespace ito