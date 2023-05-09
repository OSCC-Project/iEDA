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
 * @file sdcClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the clock command obj of sdc.
 * @version 0.1
 * @date 2020-11-22
 *
 */

#include "SdcClock.hh"

namespace ista {
SdcClock::SdcClock(const char* clock_name)
    : _clock_name(clock_name), _period(0.0) {}

SdcGenerateCLock::SdcGenerateCLock(const char* clock_name)
    : SdcClock(clock_name) {}

};  // namespace ista
