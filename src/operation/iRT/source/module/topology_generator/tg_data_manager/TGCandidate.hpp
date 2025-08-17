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

#include "PlanarCoord.hpp"
#include "RTHeader.hpp"
#include "Segment.hpp"

namespace irt {

class TGCandidate
{
 public:
  TGCandidate() = default;
  TGCandidate(int32_t topo_idx, const std::vector<Segment<PlanarCoord>>& routing_segment_list, double cost)
  {
    _topo_idx = topo_idx;
    _routing_segment_list = routing_segment_list;
    _cost = cost;
  }
  ~TGCandidate() = default;
  // getter
  int32_t get_topo_idx() const { return _topo_idx; }
  std::vector<Segment<PlanarCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  double get_cost() const { return _cost; }
  // setter
  void set_topo_idx(const int32_t topo_idx) { _topo_idx = topo_idx; }
  void set_routing_segment_list(const std::vector<Segment<PlanarCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_cost(const double cost) { _cost = cost; }
  // function

 private:
  int32_t _topo_idx = -1;
  std::vector<Segment<PlanarCoord>> _routing_segment_list;
  double _cost = 0.0;
};

}  // namespace irt
