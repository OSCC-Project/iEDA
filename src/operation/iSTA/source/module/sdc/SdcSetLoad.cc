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
 * @file SdcSetLoad.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set_load constrain implemention.
 * @version 0.1
 * @date 2021-04-14
 */
#include "SdcSetLoad.hh"

namespace ista {
SdcSetLoad::SdcSetLoad(const char* constrain_name, double load_value)
    : SdcIOConstrain(constrain_name),
      _rise(0),
      _fall(0),
      _max(0),
      _min(0),
      _pin_load(0),
      _wire_load(0),
      _subtract_pin_load(0),
      _allow_negative_load(0),
      _reserved(0),
      _load_value(load_value) {}

}  // namespace ista
