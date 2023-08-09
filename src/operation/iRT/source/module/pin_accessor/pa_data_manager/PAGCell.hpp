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

#include "PASourceType.hpp"
#include "SpaceRegion.hpp"

namespace irt {

class PAGCell : public SpaceRegion
{
 public:
  PAGCell() = default;
  ~PAGCell() = default;
  // getter
  std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>>& get_source_routing_net_rect_map()
  {
    return _source_routing_net_rect_map;
  }
  std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>>& get_source_cut_net_rect_map()
  {
    return _source_cut_net_rect_map;
  }
  std::map<PASourceType, void*>& get_source_region_query_map() { return _source_region_query_map; }
  // setter
  void set_source_routing_net_rect_map(
      const std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>>& source_routing_net_rect_map)
  {
    _source_routing_net_rect_map = source_routing_net_rect_map;
  }
  void set_source_cut_net_rect_map(
      const std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>>& source_cut_net_rect_map)
  {
    _source_cut_net_rect_map = source_cut_net_rect_map;
  }
  void set_source_region_query_map(const std::map<PASourceType, void*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  // function

 private:
  std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>> _source_routing_net_rect_map;
  std::map<PASourceType, std::map<irt_int, std::map<irt_int, std::vector<LayerRect>>>> _source_cut_net_rect_map;
  std::map<PASourceType, void*> _source_region_query_map;
};

}  // namespace irt
