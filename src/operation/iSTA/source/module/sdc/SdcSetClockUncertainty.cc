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
 * @file SdcSetClockUncertainty.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_uncertainty implemention.
 * @version 0.1
 * @date 2021-10-25
 */

#include "SdcSetClockUncertainty.hh"

namespace ista {

SdcSetClockUncertainty::SdcSetClockUncertainty(double uncertainty_value)
    : _rise(1),
      _fall(1),
      _setup(1),
      _hold(1),
      _reserved(0),
      _uncertainty_value(uncertainty_value) {}

}  // namespace ista
