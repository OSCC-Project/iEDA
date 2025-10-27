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

#include "Direction.hpp"
#include "LayerCoord.hpp"

namespace irt {

class DRGroup
{
 public:
  DRGroup() = default;
  ~DRGroup() = default;
  // getter
  std::map<LayerCoord, std::set<Direction>, CmpLayerCoordByXASC>& get_coord_direction_map() { return _coord_direction_map; }
  // setter
  void set_coord_direction_map(const std::map<LayerCoord, std::set<Direction>, CmpLayerCoordByXASC>& coord_direction_map)
  {
    _coord_direction_map = coord_direction_map;
  }
  // function

 private:
  std::map<LayerCoord, std::set<Direction>, CmpLayerCoordByXASC> _coord_direction_map;
};

}  // namespace irt
