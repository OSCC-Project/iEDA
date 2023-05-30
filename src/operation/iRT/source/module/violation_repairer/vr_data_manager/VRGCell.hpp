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

class VRGCell
{
 public:
  VRGCell() = default;
  ~VRGCell() = default;
  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_blockage_map() { return _net_blockage_map; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<PlanarRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  // function

 private:
  PlanarRect _real_rect;
  std::map<irt_int, std::vector<PlanarRect>> _net_blockage_map;
};

}  // namespace irt
