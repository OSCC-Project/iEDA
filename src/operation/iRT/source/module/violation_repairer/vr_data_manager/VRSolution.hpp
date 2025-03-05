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

#include "EXTLayerRect.hpp"
#include "LayerCoord.hpp"

namespace irt {

class VRSolution
{
 public:
  VRSolution() = default;
  ~VRSolution() = default;
  // getter
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  std::vector<EXTLayerRect>& get_routing_patch_list() { return _routing_patch_list; }
  double get_env_cost() const { return _env_cost; }
  // setter
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_routing_patch_list(const std::vector<EXTLayerRect>& routing_patch_list) { _routing_patch_list = routing_patch_list; }
  void set_env_cost(const double env_cost) { _env_cost = env_cost; }

 private:
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  std::vector<EXTLayerRect> _routing_patch_list;
  double _env_cost = 0.0;
};

struct CmpVRSolution
{
  bool operator()(const VRSolution& a, const VRSolution& b) const { return a.get_env_cost() < b.get_env_cost(); }
};

}  // namespace irt
