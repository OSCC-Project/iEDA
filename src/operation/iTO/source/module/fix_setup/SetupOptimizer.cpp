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

#include "SetupOptimizer.h"

#include "EstimateParasitics.h"
#include "Placer.h"
#include "Reporter.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "data_manager.h"
#include "liberty/Lib.hh"
#include "timing_engine.h"

using namespace std;

namespace ito {

SetupOptimizer* SetupOptimizer::_instance = nullptr;

void SetupOptimizer::performBuffering(const char* net_name)
{
  init();
  int begin_buffer_num = toDmInst->get_buffer_num();

  Netlist* design_nl = timingEngine->get_sta_engine()->get_netlist();
  ista::Net* net = design_nl->findNet(net_name);

  LOG_ERROR_IF(!net) << "Cannot find net: " << net_name << " in design.\n";

  auto driver = net->getDriver();
  Pin* driver_pin = dynamic_cast<Pin*>(driver);

  int fanout = getFanoutNumber(driver_pin);

  bool buffering_success = performBufferingIfNecessary(driver_pin, fanout);

  if (buffering_success) {
    toRptInst->get_ofstream() << "Net : " << net_name << " insert " << toDmInst->get_buffer_num() - begin_buffer_num << " buffers.\n";
  } else {
    toRptInst->get_ofstream() << "No buffers inserted in Net: " << net_name << endl;
  }
  toRptInst->get_ofstream().close();
}

void SetupOptimizer::optimizeSetup()
{
  int begin_buffer_num = toDmInst->get_buffer_num();
  int begin_resize_num = toDmInst->get_resize_num();
  // endpoints with setup violation.
  TOVertexSeq end_pts_setup_violation;

  // step 1. init
  init();
  // step 2. find violation endpoints
  checkAndFindVioaltion(end_pts_setup_violation);
  // step 3. optimization
  optimizeViolationProcess(end_pts_setup_violation);
  // step 4. report
  report(begin_buffer_num, begin_resize_num);
}

void SetupOptimizer::report(int begin_buffer_num, int begin_resize_num)
{
  reportWNSAndTNS();

  StaSeqPathData* worst_path = worstRequiredPath();
  TOSlack worst_slack = worst_path->getSlackNs();

  toRptInst->get_ofstream() << "TO: Total insert " << toDmInst->get_buffer_num() - begin_buffer_num << " buffers when fix setup.\n"
                            << "TO: Total resize " << toDmInst->get_resize_num() - begin_resize_num << " instances when fix setup.\n";
  toRptInst->get_ofstream().close();

  if (worst_slack < toConfig->get_setup_target_slack()) {
    toRptInst->get_ofstream() << "TO: Failed to fix all setup violations in current design.\n";
    toRptInst->get_ofstream().close();
  }
  if (toDmInst->reachMaxArea()) {
    toRptInst->get_ofstream() << "TO: Reach the maximum utilization of current design.\n";
    toRptInst->get_ofstream().close();
  }

  toRptInst->reportTime(false);
}

void SetupOptimizer::reportWNSAndTNS()
{
  double tns = kInf;
  double wns = kInf;
  auto clk_list = timingEngine->get_sta_engine()->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto tns1 = timingEngine->get_sta_engine()->getTNS(clk_name, AnalysisMode::kMax);
    auto wns1 = timingEngine->get_sta_engine()->getWNS(clk_name, AnalysisMode::kMax);
    tns = min(tns, tns1);
    wns = min(wns, wns1);
  }
  toRptInst->get_ofstream() << setiosflags(ios::right) << setw(20) << tns << setw(20) << wns << resetiosflags(ios::right) << endl
                            << "-----------------------------------------" << endl;
  toRptInst->get_ofstream().close();
}

}  // namespace ito
