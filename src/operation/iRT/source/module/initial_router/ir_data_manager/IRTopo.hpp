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

#include "IRGroup.hpp"

namespace irt {

class IRTopo
{
 public:
  IRTopo() = default;
  ~IRTopo() = default;
  // getter
  int32_t get_net_idx() { return _net_idx; }
  std::vector<IRGroup>& get_ir_group_list() { return _ir_group_list; }
  PlanarRect& get_bounding_box() { return _bounding_box; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  // const getter
  const std::vector<IRGroup>& get_ir_group_list() const { return _ir_group_list; }
  const PlanarRect& get_bounding_box() const { return _bounding_box; }
  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_ir_group_list(const std::vector<IRGroup>& ir_group_list) { _ir_group_list = ir_group_list; }
  void set_bounding_box(const PlanarRect& bounding_box) { _bounding_box = bounding_box; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // function

 private:
  int32_t _net_idx = -1;
  std::vector<IRGroup> _ir_group_list;
  PlanarRect _bounding_box;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
};

}  // namespace irt
