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
#include "PlanarRect.hpp"
#include "RANet.hpp"
#include "RANetNode.hpp"
#include "RASourceType.hpp"
#include "RTU.hpp"
#include "RegionQuery.hpp"

namespace irt {

class RAGCell
{
 public:
  RAGCell() = default;
  ~RAGCell() = default;
  // getter
  PlanarRect& get_base_region() { return _base_region; }
  std::map<RASourceType, RegionQuery>& get_source_region_query_map() { return _source_region_query_map; }
  irt_int get_resource_supply() const { return _resource_supply; }
  std::vector<RANetNode>& get_ra_net_node_list() { return _ra_net_node_list; }
  // setter
  void set_base_region(const PlanarRect& base_region) { _base_region = base_region; }
  void set_resource_supply(const irt_int resource_supply) { _resource_supply = resource_supply; }
  void set_ra_net_node_list(const std::vector<RANetNode>& ra_net_node_list) { _ra_net_node_list = ra_net_node_list; }
  // function
  RegionQuery& getRegionQuery(RASourceType ra_source_type) { return _source_region_query_map[ra_source_type]; }

 private:
  PlanarRect _base_region;
  std::map<RASourceType, RegionQuery> _source_region_query_map;
  irt_int _resource_supply = 0;
  std::vector<RANetNode> _ra_net_node_list;
};

}  // namespace irt
