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

#include "Guide.hpp"
#include "LayerRect.hpp"
#include "MTree.hpp"
#include "Segment.hpp"

namespace irt {

class GuideSeg : public Segment<Guide>
{
 public:
  GuideSeg() = default;
  ~GuideSeg() = default;
  // getter
  std::set<int32_t>& get_pin_idx_set() { return _pin_idx_set; }
  MTree<LayerCoord>& get_routing_tree() { return _routing_tree; }
  // setter
  void set_pin_idx_set(const std::set<int32_t>& pin_idx_set) { _pin_idx_set = pin_idx_set; }
  void set_routing_tree(const MTree<LayerCoord>& routing_tree) { _routing_tree = routing_tree; }
  // function
  bool toTA() { return get_first().get_grid_coord().get_planar_coord() != get_second().get_grid_coord().get_planar_coord(); }
  bool toDR() { return !toTA(); }

 private:
  std::set<int32_t> _pin_idx_set;
  MTree<LayerCoord> _routing_tree;
};

}  // namespace irt
