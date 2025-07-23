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
/**
 * @project		vectorization
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "vec_layer.h"

namespace ivec {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VecNet* VecLayoutLayer::get_net(int net_id)
{
  auto it = _net_map.find(net_id);
  if (it != _net_map.end()) {
    return &it->second;
  }

  return nullptr;
}

VecNet* VecLayoutLayer::getOrCreateNet(int net_id)
{
  auto* net = get_net(net_id);
  if (net == nullptr) {
    VecNet new_net(net_id);
    auto result = _net_map.insert(std::make_pair(net_id, new_net));

    if (result.first != _net_map.end()) {
      return &result.first->second;
    }
  }

  return net;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VecLayoutLayer* VecLayoutLayers::findLayoutLayer(int order)
{
  auto layout_layer = _layout_layers.find(order);
  if (layout_layer != _layout_layers.end()) {
    return &layout_layer->second;
  }

  return nullptr;
}

}  // namespace ivec
