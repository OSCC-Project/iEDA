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

#include "HoldOptimizer.h"

#include "../../config/ToConfig.h"
#include "EstimateParasitics.h"
#include "Master.h"
#include "Placer.h"
#include "Point.h"
#include "Reporter.h"
#include "data_manager.h"
#include "timing_engine.h"

using namespace std;

namespace ito {

HoldOptimizer* HoldOptimizer::_instance = nullptr;

HoldOptimizer::HoldOptimizer()
{
  _target_slack = toConfig->get_hold_target_slack();
}

void HoldOptimizer::optimizeHold()
{
  int begin_buffer_num = toDmInst->get_buffer_num();
  init();

  process();

  report(begin_buffer_num);
}

void HoldOptimizer::report(int begin_buffer_num)
{
  TOSlack worst_timing_slack_hold = timingEngine->getWorstSlack(AnalysisMode::kMin);
  if (worst_timing_slack_hold < _target_slack) {
    checkAndFindVioaltion();
    toRptInst->get_ofstream() << "\nTO: Failed to fix all hold violations in current design. There are still "
                              << _end_pts_hold_violation.size() << " endpoints with hold violation." << endl;
  }

  toRptInst->get_ofstream() << "\nTO: Finish hold optimization!\n"
                            << "TO: Total insert " << toDmInst->get_buffer_num() - begin_buffer_num
  << " buffers when fix hold.\n";

  reportWNSAndTNS();

  toRptInst->reportTime(false);
}

}  // namespace ito
