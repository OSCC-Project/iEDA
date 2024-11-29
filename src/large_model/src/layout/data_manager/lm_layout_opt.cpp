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

#include "lm_layout_opt.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "Log.hh"
#include "idm.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmLayoutOptimize::checkPinConnection()
{
  ieda::Stats stats;

  LOG_INFO << "LM check pin connections for routing layer start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& net_map = _layout->get_graph().get_net_map();

  int connected_num = 0;

  for (auto& [net_id, net] : net_map) {
    auto& pin_ids = net.get_pin_ids();
    auto& wires = net.get_wires();
    if (wires.size() <= 0) {
      continue;
    }

    std::set<int> count_pins = std::set<int>(pin_ids.begin(), pin_ids.end());
    bool b_io = false;

    for (auto& wire : wires) {
      auto& [node1, node2] = wire.get_connected_nodes();
      if (node1->get_node_data().get_pin_id() != -1) {
        count_pins.erase(node1->get_node_data().get_pin_id());
        if (node1->get_node_data().is_io()) {
          b_io = true;
        }
      }
      if (node2->get_node_data().get_pin_id() != -1) {
        count_pins.erase(node2->get_node_data().get_pin_id());
        if (node2->get_node_data().is_io()) {
          b_io = true;
        }
      }
    }

    if (count_pins.size() > 0) {
      for (auto pin_id : count_pins) {
        LOG_INFO << "disconnected pin : " << pin_id;
        reconnectPin(net, pin_id);
      }

      if (b_io) {
        LOG_INFO << "has io... ";
      }
      LOG_INFO << "net " << net_id << " reconnect : " << count_pins.size();
    } else {
      LOG_INFO << "net connected " << net_id;
      connected_num++;
    }
  }

  LOG_INFO << "net reconnect ratio " << connected_num << " / " << net_map.size();

  omp_destroy_lock(&lck);

  LOG_INFO << "LM check pin connections for routing layer end...";
}

void LmLayoutOptimize::reconnectPin(LmNet& lm_net, int pin_id)
{
  auto& patch_layers = _layout->get_patch_layers();

  auto* lm_pin = lm_net.get_pin(pin_id);
  if (lm_pin) {
    auto& shape_map = lm_pin->get_shape_map();
    for (auto& [layer_id, layer_shape] : shape_map) {
      auto* patch_layer = patch_layers.findPatchLayer(layer_id);
      if (nullptr == patch_layer) {
        LOG_WARNING << "Can not get layer order : " << layer_id;
        continue;
      }
      auto& grid = patch_layer->get_grid();
      auto& node_matrix = grid.get_node_matrix();
      for (auto& rect : layer_shape.rect_list) {
        for (int row = rect.row_start; row <= rect.row_end; ++row) {
          for (int col = rect.col_start; col <= rect.col_end; ++col) {
            auto& node_data = node_matrix[row][col].get_node_data();
            if (node_data.is_connected() || node_data.is_connecting()) {
              int a = 0;
              a += 1;
            }

            if (node_data.is_enclosure() || node_data.is_delta() || node_data.is_wire()) {
              int a = 0;
              a += 1;
            }
          }
        }
      }
    }
  }
}

void LmLayoutOptimize::wirePruning()
{
  ieda::Stats stats;

  LOG_INFO << "LM optimize connections for routing layer start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& net_map = _layout->get_graph().get_net_map();

  uint64_t total = 0;
  uint64_t pruning_total = 0;

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < net_map.size(); ++i) {
    //   for (auto& [net_id, net] : net_map) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto net_id = it->first;
    auto& net = it->second;

    std::vector<LmNode*> node_list = rebuildGridNode(net);

    uint64_t net_pruning_num = 0;

    uint64_t pruning_num = 0;
    while ((pruning_num = pruningNode(node_list)) > 0) {
      net_pruning_num += pruning_num;
    }

    /// remove redundant node
    int remove_size = removeRedundancy(node_list);

    /// process ring
    while (processRing(node_list) > 0) {
    }

    while ((pruning_num = pruningNode(node_list)) > 0) {
      net_pruning_num += pruning_num;
    }

    /// remove redundant node
    remove_size = remove_size + removeRedundancy(node_list);

    /// rebuild wires
    rebuildGraph(node_list, net);

    omp_set_lock(&lck);
    pruning_total = pruning_total + net_pruning_num + remove_size;
    total += node_list.size();
    omp_unset_lock(&lck);

    LOG_INFO << "net " << net_id << " pruning " << node_list.size() << " -> " << net_pruning_num + remove_size;
  }

  LOG_INFO << "net pruning ratio " << pruning_total << " / " << total;

  omp_destroy_lock(&lck);

  LOG_INFO << "LM optimize connections for routing layer end...";
}

int LmLayoutOptimize::pruningNode(std::vector<LmNode*>& node_list)
{
  std::sort(node_list.begin(), node_list.end(), [](LmNode* a, LmNode* b) { return a->getconnected_num() < b->getconnected_num(); });

  std::unordered_set<LmNode*> redundant_list;

  for (int i = 0; i < (int) node_list.size(); ++i) {
    auto* node = node_list[i];
    if (needPruning(node)) {
      redundant_list.insert(node);

      if (node->left != nullptr) {
        node->left->right = nullptr;
      }

      if (node->right != nullptr) {
        node->right->left = nullptr;
      }

      if (node->up != nullptr) {
        node->up->down = nullptr;
      }

      if (node->down != nullptr) {
        node->down->up = nullptr;
      }
    }
  }

  node_list.erase(std::remove_if(node_list.begin(), node_list.end(),
                                 [&redundant_list](LmNode* remove_node) {
                                   return std::find(redundant_list.begin(), redundant_list.end(), remove_node) != redundant_list.end();
                                 }),
                  node_list.end());

  return redundant_list.size();
}

int LmLayoutOptimize::processRing(std::vector<LmNode*>& node_list)
{
  auto can_remove = [](LmNode* node) -> bool {
    if (node->get_node_data().is_via()) {
      return false;
    }

    int layer_node_num = 0;
    for (auto* neighbout_node : {node->left, node->right, node->up, node->down}) {
      layer_node_num += (neighbout_node != nullptr ? 1 : 0);
    }

    return layer_node_num < 3 ? true : false;
  };

  auto has_ring = [&](LmNode* node) -> LmNode* {
    if (node->up != nullptr && node->up->right != nullptr && node->up->right->down != nullptr && node->up->right->down->left != nullptr) {
      if (can_remove(node)) {
        return node;
      }

      if (can_remove(node->up)) {
        return node->up;
      }

      if (can_remove(node->up->right)) {
        return node->up->right;
      }

      if (can_remove(node->up->right->down)) {
        return node->up->right->down;
      }
    }
    return nullptr;
  };

  std::sort(node_list.begin(), node_list.end(), [](LmNode* a, LmNode* b) { return a->getconnected_num() > b->getconnected_num(); });

  std::unordered_set<LmNode*> redundant_list;

  for (int i = 0; i < (int) node_list.size(); ++i) {
    auto* ring_node = has_ring(node_list[i]);
    if (ring_node != nullptr) {
      redundant_list.insert(ring_node);

      if (ring_node->left != nullptr) {
        ring_node->left->right = nullptr;
      }

      if (ring_node->right != nullptr) {
        ring_node->right->left = nullptr;
      }

      if (ring_node->up != nullptr) {
        ring_node->up->down = nullptr;
      }

      if (ring_node->down != nullptr) {
        ring_node->down->up = nullptr;
      }
    }
  }

  node_list.erase(std::remove_if(node_list.begin(), node_list.end(),
                                 [&redundant_list](LmNode* remove_node) {
                                   return std::find(redundant_list.begin(), redundant_list.end(), remove_node) != redundant_list.end();
                                 }),
                  node_list.end());

  return redundant_list.size();
}

int LmLayoutOptimize::removeRedundancy(std::vector<LmNode*>& node_list)
{
  auto do_remove = [](LmNode* node) -> bool {
    if (node->get_node_data().is_via()) {
      return false;
    }

    /// horizontal
    if (node->left != nullptr && node->right != nullptr && node->up == nullptr && node->down == nullptr) {
      node->left->right = node->right;
      node->right->left = node->left;
      return true;
    }

    /// vertical
    if (node->left == nullptr && node->right == nullptr && node->up != nullptr && node->down != nullptr) {
      node->up->down = node->down;
      node->down->up = node->up;
      return true;
    }

    return false;
  };

  std::unordered_set<LmNode*> redundant_list;
  std::vector<bool> visited_list(node_list.size(), false);
  for (int i = 0; i < (int) node_list.size(); ++i) {
    if (visited_list[i]) {
      continue;
    }

    visited_list[i] = true;

    auto* node = node_list[i];
    if (true == do_remove(node)) {
      redundant_list.insert(node);
    }
  }

  node_list.erase(std::remove_if(node_list.begin(), node_list.end(),
                                 [&redundant_list](LmNode* remove_node) {
                                   return std::find(redundant_list.begin(), redundant_list.end(), remove_node) != redundant_list.end();
                                 }),
                  node_list.end());

  return redundant_list.size();
}

std::vector<LmNode*> LmLayoutOptimize::rebuildGridNode(LmNet& lm_net)
{
  std::set<LmNode*> node_list;
  for (auto& wire : lm_net.get_wires()) {
    for (auto& [node1, node2] : wire.get_paths()) {
      if (node1->get_layer_id() == node2->get_layer_id()) {
        if ((node1->get_col_id() == node2->get_col_id()) && node1->get_row_id() == node2->get_row_id()) {
          int a = 0;
          a += 1;
        }
        /// same layer
        if (node1->get_row_id() == node2->get_row_id()) {
          if (node1->get_col_id() > node2->get_col_id()) {
            node1->left = node2;
            node2->right = node1;
          } else if (node1->get_col_id() < node2->get_col_id()) {
            node1->right = node2;
            node2->left = node1;
          } else {
          }
        }

        if (node1->get_col_id() == node2->get_col_id()) {
          if (node1->get_row_id() > node2->get_row_id()) {
            node1->down = node2;
            node2->up = node1;
          } else if (node1->get_row_id() < node2->get_row_id()) {
            node1->up = node2;
            node2->down = node1;
          } else {
          }
        }
      } else {
        if (node1->get_layer_id() > node2->get_layer_id()) {
          node1->bottom = node2;
          node2->top = node1;
        } else {
          node1->top = node2;
          node2->bottom = node1;
        }
      }

      node1->get_node_data().reset_visited();
      node2->get_node_data().reset_visited();

      node_list.insert(node1);
      node_list.insert(node2);
    }
  }

  return std::vector<LmNode*>(node_list.begin(), node_list.end());
}

bool LmLayoutOptimize::needPruning(LmNode* node)
{
  if (node->get_node_data().is_via()) {
    return false;
  }

  int layer_node_num = 0;
  for (auto* neighbout_node : {node->left, node->right, node->up, node->down}) {
    layer_node_num += (neighbout_node != nullptr ? 1 : 0);
  }

  return layer_node_num < 2 ? true : false;
}

// void LmLayoutOptimize::rebuildGraph(std::vector<LmNode*>& node_list, LmNet& lm_net)
// {
//   auto need_connected = [](LmNode* node) -> bool {
//     if (node->get_node_data().is_via()) {
//       return true;
//     }

//     int layer_node_num = 0;
//     for (auto* neighbout_node : {node->left, node->right, node->up, node->down}) {
//       layer_node_num += (neighbout_node != nullptr ? 1 : 0);
//     }

//     return layer_node_num > 2 || layer_node_num == 1 ? true : false;
//   };

//   auto build_wire = [&](LmNode* node1, LmNode* node2, LmNet& lm_net) -> bool {
//     if (node2->get_node_data().is_visited()) {
//       return false;
//     }

//     if (need_connected(node2)) {
//       LmNetWire wire(node1, node2);
//       wire.add_path(node1, node2);
//       lm_net.addWire(wire);
//       return true;
//     } else {
//       /// path
//       LmNetWire wire;
//       wire.set_start(node1);

//       auto start = node1;
//       auto end = node2;
//       //   while (false == end->get_node_data().is_visited() && false == need_connected(end)) {
//       while (false == need_connected(end)) {
//         /// if path, set visited
//         end->get_node_data().set_visited();
//         wire.add_path(start, end);

//         for (auto* end_neighbor : {end->bottom, end->top, end->left, end->right, end->up, end->down}) {
//           if (end_neighbor != nullptr && end_neighbor != start) {
//             start = end;
//             end = end_neighbor;
//             break;
//           }
//         }
//       }

//       wire.set_end(end);
//       wire.add_path(start, end);
//       lm_net.addWire(wire);
//       return true;
//     }

//     return false;
//   };

//   int origin_wire_size = lm_net.get_wires().size();
//   lm_net.clearWire();

//   for (auto* node : node_list) {
//     if (node->get_node_data().is_visited()) {
//       continue;
//     }

//     if (need_connected(node)) {
//       for (auto* neighbout_node : {node->bottom, node->top, node->left, node->right, node->up, node->down}) {
//         if (nullptr != neighbout_node && false == neighbout_node->get_node_data().is_visited()) {
//           build_wire(node, neighbout_node, lm_net);
//         }
//       }

//       /// set visited
//       node->get_node_data().set_visited();
//     }
//   }

//   LOG_INFO << "net " << lm_net.get_net_id() << " wire size " << origin_wire_size << " -> " << lm_net.get_wires().size();
// }

void LmLayoutOptimize::rebuildGraph(std::vector<LmNode*>& node_list, LmNet& lm_net)
{
  auto need_connected = [](LmNode* node) -> bool {
    if (node->get_node_data().is_via()) {
      return true;
    }

    int layer_node_num = 0;
    for (auto* neighbout_node : {node->left, node->right, node->up, node->down}) {
      layer_node_num += (neighbout_node != nullptr ? 1 : 0);
    }

    return layer_node_num > 2 || layer_node_num == 1 ? true : false;
  };

  auto build_wire = [&](LmNode* node1, LmNode* node2, LmNet& lm_net) -> bool {
    if (node2->get_node_data().is_visited()) {
      return false;
    }

    LmNetWire wire(node1, node2);
    wire.add_path(node1, node2);
    lm_net.addWire(wire);
    return true;
  };

  int origin_wire_size = lm_net.get_wires().size();
  lm_net.clearWire();

  for (auto* node : node_list) {
    if (node->get_node_data().is_visited()) {
      continue;
    }

    // if (need_connected(node)) {
    for (auto* neighbout_node : {node->bottom, node->top, node->left, node->right, node->up, node->down}) {
      if (nullptr != neighbout_node) {
        build_wire(node, neighbout_node, lm_net);
      }
    }

    /// set visited
    node->get_node_data().set_visited();
  }

  LOG_INFO << "net " << lm_net.get_net_id() << " wire size " << origin_wire_size << " -> " << lm_net.get_wires().size();
}

}  // namespace ilm