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
 * @file SdcSetClockLatency.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_uncertainty implemention.
 * @version 0.1
 * @date 2021-10-21
 */
#include "SdcSetClockLatency.hh"

namespace ista {

SdcSetClockLatency::SdcSetClockLatency(double delay_value)
    : _rise(0),
      _fall(0),
      _max(0),
      _min(0),
      _early(0),
      _late(0),
      _reserved(0),
      _delay_value(delay_value) {}

}  // namespace ista
