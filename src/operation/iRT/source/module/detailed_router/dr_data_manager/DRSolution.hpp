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

class DRSolution
{
 public:
  DRSolution() = default;
  ~DRSolution() = default;
  // getter
  bool get_is_only() const { return _is_only; }
  EXTLayerRect& get_curr_patch() { return _curr_patch; }
  std::vector<EXTLayerRect>& get_routing_patch_list() { return _routing_patch_list; }
  // setter
  void set_is_only(const bool is_only) { _is_only = is_only; }
  void set_curr_patch(const EXTLayerRect& curr_patch) { _curr_patch = curr_patch; }
  void set_routing_patch_list(const std::vector<EXTLayerRect>& routing_patch_list) { _routing_patch_list = routing_patch_list; }

 private:
  bool _is_only = false;
  EXTLayerRect _curr_patch;
  std::vector<EXTLayerRect> _routing_patch_list;
};

}  // namespace irt
