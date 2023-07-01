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

namespace irt {

class PAGCell : public LayerCoord
{
 public:
  PAGCell() = default;
  ~PAGCell() = default;

  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<irt_int, std::vector<LayerRect>>& get_net_blockage_map() { return _net_blockage_map; }
  void* get_net_blockage_region_query() { return _net_blockage_region_query; }
  std::map<irt_int, std::vector<LayerRect>>& get_net_enclosure_map() { return _net_enclosure_map; }
  void* get_net_enclosure_region_query() { return _net_enclosure_region_query; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<LayerRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  void set_net_blockage_region_query(void* net_blockage_region_query) { _net_blockage_region_query = net_blockage_region_query; }
  void set_net_enclosure_map(const std::map<irt_int, std::vector<LayerRect>>& net_enclosure_map) { _net_enclosure_map = net_enclosure_map; }
  void set_net_enclosure_region_query(void* net_enclosure_region_query) { _net_enclosure_region_query = net_enclosure_region_query; }
  // function

 private:
  PlanarRect _real_rect;
  std::map<irt_int, std::vector<LayerRect>> _net_blockage_map;
  void* _net_blockage_region_query;
  std::map<irt_int, std::vector<LayerRect>> _net_enclosure_map;
  void* _net_enclosure_region_query;
};

}  // namespace irt
