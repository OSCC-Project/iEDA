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
#include "vec_patch.h"

#include "vec_net.h"

namespace ivec {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VecPatchLayer::addSubnet(int net_id, int64_t wire_id, VecNode* node1, VecNode* node2)
{
  auto* sub_net = findNet(net_id);
  if (sub_net == nullptr) {
    VecNet vec_net(net_id);
    VecNetWire wire(nullptr, nullptr, wire_id);
    wire.add_path(node1, node2);
    vec_net.addWire(wire);

    _sub_nets.insert(std::make_pair(net_id, vec_net));
  } else {
    auto* wire = sub_net->findWire(wire_id);
    if (wire == nullptr) {
      VecNetWire wire(nullptr, nullptr, wire_id);
      wire.add_path(node1, node2);
      sub_net->addWire(wire);
    } else {
      wire->add_path(node1, node2);
    }
  }
}

void VecPatchLayer::addSubnet(VecNet* sub_net, int64_t wire_id, VecNode* node1, VecNode* node2)
{
  if (sub_net != nullptr) {
    auto* wire = sub_net->findWire(wire_id);
    if (wire == nullptr) {
      VecNetWire wire(nullptr, nullptr, wire_id);
      wire.add_path(node1, node2);
      sub_net->addWire(wire);
    } else {
      wire->add_path(node1, node2);
    }
  }
}

VecNet* VecPatchLayer::findNet(int net_id)
{
  auto it = _sub_nets.find(net_id);
  if (it != _sub_nets.end()) {
    return &it->second;
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VecPatchLayer* VecPatch::findLayer(int layer_id)
{
  auto it = _layer_map.find(layer_id);
  if (it != _layer_map.end()) {
    return &it->second;
  }

  return nullptr;
}

}  // namespace ivec
