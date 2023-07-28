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

#include "../../../../database/interaction/ids.hpp"
#include "LayerRect.hpp"
#include "RQShape.hpp"
#include "RTU.hpp"

namespace irt {

using StaticBox = std::pair<BoostBox, RQShape*>;

class RegionQuery
{
 public:
  RegionQuery() {}
  ~RegionQuery() {}
  // getter
  std::map<irt_int, std::vector<RQShape>>& get_obj_id_shape_map() { return _obj_id_shape_map; }
  std::map<irt_int, bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16>>>& get_region_map() { return _region_map; }
  // setters
  // function

 private:
  std::map<irt_int, std::vector<RQShape>> _obj_id_shape_map;
  std::map<irt_int, bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16>>> _region_map;
};

}  // namespace irt
