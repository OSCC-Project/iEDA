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
  std::map<VRSourceType, RegionQuery*>& get_source_region_query_map() { return _source_region_query_map; }
  // setter
  void set_vr_gcell_id(const VRGCellId& vr_gcell_id) { _vr_gcell_id = vr_gcell_id; }
  void set_source_region_query_map(const std::map<VRSourceType, RegionQuery*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  // function
  RegionQuery* getRegionQuery(VRSourceType vr_source_type)
  {
    RegionQuery*& region_query = _source_region_query_map[vr_source_type];
    if (region_query == nullptr) {
      region_query = DC_INST.initRegionQuery();
    }
    return region_query;
  }

 private:
  VRGCellId _vr_gcell_id;
  std::map<VRSourceType, RegionQuery*> _source_region_query_map;
};

}  // namespace irt
