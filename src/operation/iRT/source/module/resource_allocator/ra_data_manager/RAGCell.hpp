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

#include "PlanarRect.hpp"
#include "RANet.hpp"
#include "RANetNode.hpp"
#include "RTU.hpp"

namespace irt {

class RAGCell
{
 public:
  RAGCell() = default;
  ~RAGCell() = default;
  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<irt_int, std::vector<PlanarRect>>& get_layer_blockage_map() { return _layer_blockage_map; }
  irt_int get_public_track_supply() const { return _public_track_supply; }
  std::vector<RANetNode>& get_ra_net_node_list() { return _ra_net_node_list; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_public_track_supply(const irt_int public_track_supply) { _public_track_supply = public_track_supply; }
  void set_ra_net_node_list(const std::vector<RANetNode>& ra_net_node_list) { _ra_net_node_list = ra_net_node_list; }
  // function

 private:
  PlanarRect _real_rect;
  std::map<irt_int, std::vector<PlanarRect>> _layer_blockage_map;
  irt_int _public_track_supply = 0;
  std::vector<RANetNode> _ra_net_node_list;
};

}  // namespace irt
