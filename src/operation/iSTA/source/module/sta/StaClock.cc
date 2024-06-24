// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of sta clock.
 * @version 0.1
 * @date 2021-02-17
 */

#include "StaClock.hh"

#include <utility>

#include "StaFunc.hh"

namespace ista {

StaClock::StaClock(const char* clock_name, ClockType clock_type, int period)
    : _clock_name(clock_name), _clock_type(clock_type), _period(period) {}

unsigned StaClock::exec(StaFunc& func) { return func(this); }

}  // namespace ista