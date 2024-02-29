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
 * @file SdcSetIODelay.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of set_input_delay and set_output_delay constrain.
 * @version 0.1
 * @date 2021-05-24
 */

#include "SdcSetIODelay.hh"

namespace ista {

SdcSetIODelay::SdcSetIODelay(const char* constrain_name, const char* clock_name,
                             double delay_value)
    : SdcIOConstrain(constrain_name),
      _rise(1),
      _fall(1),
      _max(1),
      _min(1),
      _clock_fall(0),
      _add(0),
      _reserved(0),
      _clock_name(clock_name),
      _delay_value(delay_value) {}

SdcSetInputDelay::SdcSetInputDelay(const char* constrain_name,
                                   const char* clock_name, double delay_value)
    : SdcSetIODelay(constrain_name, clock_name, delay_value) {}

SdcSetOutputDelay::SdcSetOutputDelay(const char* constrain_name,
                                     const char* clock_name, double delay_value)
    : SdcSetIODelay(constrain_name, clock_name, delay_value) {}

}  // namespace ista