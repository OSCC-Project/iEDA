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

#include "ViolatedPin.hpp"

namespace irt {

class ViolatedGroup
{
 public:
  ViolatedGroup() = default;
  ViolatedGroup(const std::vector<ViolatedPin>& violated_pin_list, const LayerCoord& balanced_coord)
  {
    _violated_pin_list = violated_pin_list;
    _balanced_coord = balanced_coord;
  }
  ~ViolatedGroup() = default;
  // getter
  std::vector<ViolatedPin>& get_violated_pin_list() { return _violated_pin_list; }
  LayerCoord& get_balanced_coord() { return _balanced_coord; }
  // setter
  void set_violated_pin_list(const std::vector<ViolatedPin>& violated_pin_list) { _violated_pin_list = violated_pin_list; }
  void set_balanced_coord(const LayerCoord& balanced_coord) { _balanced_coord = balanced_coord; }
  // function

 private:
  std::vector<ViolatedPin> _violated_pin_list;
  LayerCoord _balanced_coord;
};

}  // namespace irt
