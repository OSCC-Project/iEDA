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

#include "SdcTimingDRC.hh"

namespace ista {

SetMaxTransition::SetMaxTransition(double transition_value)
    : SdcTimingDRC(transition_value),
      _is_clock_path(0),
      _is_data_path(0),
      _is_rise(0),
      _is_fall(0) {}

SetMaxCapacitance::SetMaxCapacitance(double capacitance_value)
    : SdcTimingDRC(capacitance_value),
      _is_clock_path(0),
      _is_data_path(0),
      _is_rise(0),
      _is_fall(0) {}

SetMaxFanout::SetMaxFanout(double fanout_value) : SdcTimingDRC(fanout_value) {}

}  // namespace ista
