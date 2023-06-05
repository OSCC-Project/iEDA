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

#include "LayerCoord.hpp"
#include "Orientation.hpp"

namespace irt {

class DRGroup
{
 public:
  DRGroup() = default;
  ~DRGroup() = default;
  // getter
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC>& get_coord_orientation_map() { return _coord_orientation_map; }
  // setter
  void set_coord_orientation_map(const std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC>& coord_orientation_map)
  {
    _coord_orientation_map = coord_orientation_map;
  }
  // function

 private:
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> _coord_orientation_map;
};

}  // namespace irt
