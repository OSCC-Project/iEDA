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
#pragma once

#include "PAPin.hpp"

namespace irt {

class ViolatedPin
{
 public:
  ViolatedPin() = default;
  ViolatedPin(const irt_int net_idx, PAPin* pa_pin, const LayerCoord& balanced_coord)
  {
    _net_idx = net_idx;
    _pa_pin = pa_pin;
    _balanced_coord = balanced_coord;
  }
  ~ViolatedPin() = default;
  // getter
  irt_int get_net_idx() const { return _net_idx; }
  PAPin* get_pa_pin() { return _pa_pin; }
  LayerCoord& get_balanced_coord() { return _balanced_coord; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_pa_pin(PAPin* pa_pin) { _pa_pin = pa_pin; }
  void set_balanced_coord(const LayerCoord& balanced_coord) { _balanced_coord = balanced_coord; }
  // function

 private:
  irt_int _net_idx;
  PAPin* _pa_pin;
  LayerCoord _balanced_coord;
};

}  // namespace irt
