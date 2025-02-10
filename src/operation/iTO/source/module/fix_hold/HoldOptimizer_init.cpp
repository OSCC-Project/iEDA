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

void HoldOptimizer::init()
{
  initBufferCell();
  LOG_ERROR_IF(_available_buffer_cells.empty()) << "Can not found specified buffers.\n";
  calcBufferCap();

  if (!_has_estimate_all_net) {
    toEvalInst->estimateAllNetParasitics();
    _has_estimate_all_net = true;
  } else {
    toEvalInst->excuteParasiticsEstimate();
  }

  timingEngine->get_sta_engine()->updateTiming();
  timingEngine->set_eval_data();

  reportWNSAndTNS();

  _all_end_points = getEndPoints();

  _hold_insert_buf_cell = ensureInsertBufferSize();

  float max_insert_instance_percent = toConfig->get_max_insert_instance_percent();
  int instance_num = timingEngine->get_sta_engine()->get_netlist()->getInstanceNum();
  _max_numb_insert_buf = max_insert_instance_percent * instance_num;
}

}  // namespace ito
