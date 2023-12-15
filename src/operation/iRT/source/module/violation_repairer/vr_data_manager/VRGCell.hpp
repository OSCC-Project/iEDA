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

#include "DRCChecker.hpp"
#include "RegionQuery.hpp"
#include "SpaceRegion.hpp"
#include "VRGCellId.hpp"
#include "VRSourceType.hpp"

namespace irt {

class VRGCell : public SpaceRegion
{
 public:
  VRGCell() = default;
  ~VRGCell() = default;
  // getter
  VRGCellId& get_vr_gcell_id() { return _vr_gcell_id; }
  std::map<VRSourceType, RegionQuery>& get_source_region_query_map() { return _source_region_query_map; }
  std::map<irt_int, irt_int>& get_layer_resource_supply_map() { return _layer_resource_supply_map; }
  std::map<irt_int, irt_int>& get_layer_resource_demand_map() { return _layer_resource_demand_map; }
  // setter
  void set_vr_gcell_id(const VRGCellId& vr_gcell_id) { _vr_gcell_id = vr_gcell_id; }
  // function
  RegionQuery& getRegionQuery(VRSourceType vr_source_type) { return _source_region_query_map[vr_source_type]; }

 private:
  VRGCellId _vr_gcell_id;
  std::map<VRSourceType, RegionQuery> _source_region_query_map;
  std::map<irt_int, irt_int> _layer_resource_supply_map;
  std::map<irt_int, irt_int> _layer_resource_demand_map;
};

}  // namespace irt
