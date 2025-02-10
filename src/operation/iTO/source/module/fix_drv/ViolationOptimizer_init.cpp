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
#include "Master.h"
#include "Placer.h"
#include "Reporter.h"
#include "ToConfig.h"
#include "ViolationOptimizer.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "timing_engine.h"

namespace ito {

bool ViolationOptimizer::init()
{
  /// init buffer
  if (false == initBuffer()) {
    LOG_ERROR_IF(!_insert_buffer_cell) << "Can not found specified buffer.\n";
    return false;
  }

  // update timing
  if (!_has_estimate_all_net) {
    toEvalInst->estimateAllNetParasitics();
    _has_estimate_all_net = true;
  } else {
    toEvalInst->excuteParasiticsEstimate();
  }
  timingEngine->get_sta_engine()->updateTiming();

  return true;
}

}  // namespace ito