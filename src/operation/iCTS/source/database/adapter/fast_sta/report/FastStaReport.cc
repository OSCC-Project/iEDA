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
/**
 * @file FastStaReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Focused diagnostics implementation for CTS fast STA.
 */

#include "FastStaReport.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "clock_state/FastStaClockState.hh"

namespace icts {

auto FastStaReport::logClockSummary(const FastStaClockContext& context) -> void
{
  LOG_INFO << "Fast STA clock \"" << context.clock_name << "\": nodes=" << context.nodes.size() << " nets=" << context.nets.size()
           << " skew=" << context.skew.skew_ns << "ns"
           << " power=" << context.power.total_power_w << "W";
}

}  // namespace icts
