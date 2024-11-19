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

void SetupOptimizer::init() {
  initBufferCell();
  LOG_ERROR_IF(_available_lib_cell_sizes.empty()) << "Can not found specified buffers.\n";

  if (!_has_estimate_all_net) {
    toEvalInst->estimateAllNetParasitics();
    _has_estimate_all_net = true;
  } else {
    toEvalInst->excuteParasiticsEstimate();
  }

  timingEngine->get_sta_engine()->updateTiming();
  timingEngine->set_eval_data();
}

void SetupOptimizer::initBufferCell()
{
  bool not_specified_buffer = toConfig->get_setup_insert_buffers().empty();
  if (not_specified_buffer) {
    TOLibertyCellSeq buf_cells = timingEngine->get_buffer_cells();
    for (auto buf : buf_cells) {
      if (strstr(buf->get_cell_name(), "CLK") == NULL) {
        _available_lib_cell_sizes.emplace_back(buf);
      }
    }
    return;
  }

  auto bufs = toConfig->get_setup_insert_buffers();
  for (auto buf : bufs) {
    auto buffer = timingEngine->get_sta_engine()->findLibertyCell(buf.c_str());
    if (!buffer) {
      LOG_INFO << "Buffer cell " << buf.c_str() << " not found" << endl;
    } else {
      _available_lib_cell_sizes.emplace_back(buffer);
    }
  }
}

StaSeqPathData* SetupOptimizer::worstRequiredPath()
{
  vector<TransType> rise_fall = {TransType::kRise, TransType::kFall};
  StaSeqPathData* worst_path = nullptr;
  TOSlack wns = kInf;
  for (auto rf : rise_fall) {
    auto path = timingEngine->get_sta_engine()->getWorstSeqData(AnalysisMode::kMax, rf);
    if (path->getSlackNs() < wns) {
      wns = path->getSlackNs();
      worst_path = path;
    }
  }
  return worst_path;
}

TOVertexSet SetupOptimizer::getEndPoints()
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

void SetupOptimizer::checkAndFindVioaltion(TOVertexSeq& end_pts_setup_violation)
{
  auto end_points = getEndPoints();

  findEndpointsWithSetupViolation(end_points, end_pts_setup_violation);
  // 根据slack进行升序排序
  sort(end_pts_setup_violation.begin(), end_pts_setup_violation.end(), [](StaVertex* end1, StaVertex* end2) {
    return end1->getWorstSlackNs(AnalysisMode::kMax) < end2->getWorstSlackNs(AnalysisMode::kMax);
  });

  toRptInst->get_ofstream() << "TO: Total find " << (int) end_pts_setup_violation.size()
                            << " endpoints with setup violation in current design.\n";
  toRptInst->get_ofstream().close();
}

void SetupOptimizer::findEndpointsWithSetupViolation(TOVertexSet  end_points,
                                                     TOVertexSeq &setup_violations) {
  setup_violations.clear();

  for (auto *end : end_points) {
    TOSlack slack = timingEngine->getWorstSlack(end, AnalysisMode::kMax);
    if (slack < toConfig->get_setup_target_slack()) {
      setup_violations.emplace_back(end);
    }
  }
}


int SetupOptimizer::getFanoutNumber(Pin* pin)
{
  auto* net = pin->get_net();
  return net->getFanouts();
}

bool SetupOptimizer::netConnectToOutputPort(Net* net)
{
  DesignObject* pin;
  FOREACH_NET_PIN(net, pin)
  {
    if (pin->isOutput()) {
      return true;
    }
  }
  return false;
}

bool SetupOptimizer::netConnectToPort(Net* net)
{
  auto load_pin_ports = net->getLoads();
  for (auto pin_port : load_pin_ports) {
    if (pin_port->isPort()) {
      return true;
    }
  }
  return false;
}

}  // namespace ito