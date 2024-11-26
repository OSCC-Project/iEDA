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

#include "lm_layout_dm.h"

#include "Log.hh"
#include "lm_layout_check.hh"
#include "lm_layout_file.h"
#include "lm_layout_init.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {
bool LmLayoutDataManager::buildLayoutData(const std::string path)
{
  init();
  buildPatchs();

  return true;
}

bool LmLayoutDataManager::buildGraphData(const std::string path)
{
  init();

  buildNetWires(true);
  if (get_graph().size() > 0) {
    // connectiviy check
    LmLayoutChecker checker;
    LOG_ERROR_IF(!checker.checkLayout(get_graph())) << "Graph is not connected";
    /// save to path
    LmLayoutFileIO file_io;
    return file_io.saveJson(path, get_graph());
  }
  return false;
}

void LmLayoutDataManager::init()
{
  LmLayoutInit layout_init(&_layout);
  layout_init.init();
}

void LmLayoutDataManager::add_net_wire(int net_id, LmNetWire wire)
{
  auto& [start, end] = wire.get_connected_nodes();
  if (start == nullptr || end == nullptr) {
    LOG_INFO << "wire error";
  }

  auto it = get_graph().find(net_id);
  if (it != get_graph().end()) {
    it->second.addWire(wire);
  } else {
    LmNet lm_net(net_id);
    lm_net.addWire(wire);
    auto result = get_graph().insert(std::make_pair(net_id, lm_net));
  }
}

std::map<int, LmNet> LmLayoutDataManager::buildNetWires(bool b_graph)
{
  ieda::Stats stats;

  LOG_INFO << "LM build net wires start...";

  int wire_num = 0;

  auto& patch_layers = _layout.get_patch_layers();
  for (auto& [layer_id, patch_layer] : patch_layers.get_patch_layer_map()) {
    if (patch_layer.is_routing()) {
      /// metal
      wire_num += buildRoutingLayer(layer_id, patch_layer);
    } else {
      /// via
      wire_num += buildCutLayer(layer_id, patch_layer);
    }
  }

  LOG_INFO << "Net number : " << get_graph().size();
  LOG_INFO << "Wire number : " << wire_num;

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM build net wires end...";

  return get_graph();
}

int LmLayoutDataManager::buildCutLayer(int layer_id, LmPatchLayer& patch_layer)
{
  int wire_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& grid = patch_layer.get_grid();
  auto& node_matrix = grid.get_node_matrix();

  if (false == patch_layer.is_routing()) {
    /// via
    auto& patch_layers = _layout.get_patch_layers();

    auto* layer_top = patch_layers.findPatchLayer(layer_id + 1);
    auto* layer_bottom = patch_layers.findPatchLayer(layer_id - 1);
    auto& node_matrix_top = layer_top->get_grid().get_node_matrix();
    auto& node_matrix_bottom = layer_bottom->get_grid().get_node_matrix();
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < grid.get_info().node_row_num; ++row) {
      for (int col = 0; col < grid.get_info().node_col_num; ++col) {
        auto& node_data = node_matrix[row][col].get_node_data();
        if (false == node_data.is_net()) {
          continue;
        }

        int net_id = node_data.get_net_id();
        /// skip node if node is not net
        if (net_id < 0) {
          continue;
        }

        omp_set_lock(&lck);
        LmNetWire wire(&node_matrix_bottom[row][col], &node_matrix_top[row][col]);
        wire.add_path(&node_matrix_bottom[row][col], &node_matrix_top[row][col]);
        add_net_wire(net_id, wire);
        ++wire_num;
        omp_unset_lock(&lck);
      }
      if (row % 1000 == 0) {
        LOG_INFO << "Patch layer " << layer_id << " Read rows : " << row << " / " << grid.get_info().node_row_num;
      }
    }

    LOG_INFO << "Patch layer " << layer_id << " Read rows : " << grid.get_info().node_row_num << " / " << grid.get_info().node_row_num;
  }

  omp_destroy_lock(&lck);

  return wire_num;
}

int LmLayoutDataManager::buildRoutingLayer(int layer_id, LmPatchLayer& patch_layer)
{
  int wire_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& grid = patch_layer.get_grid();
  auto& node_matrix = grid.get_node_matrix();

  if (true == patch_layer.is_routing()) {
    for (int row = 0; row < grid.get_info().node_row_num; ++row) {
      for (int col = 0; col < grid.get_info().node_col_num; ++col) {
        if (node_matrix[row][col].get_node_data().is_visited() || false == node_matrix[row][col].get_node_data().is_net()) {
          /// skip node
          continue;
        }

        auto& node_data = node_matrix[row][col].get_node_data();
        int net_id = node_data.get_net_id();
        /// skip node if node is not net
        if (net_id < 0) {
          node_matrix[row][col].get_node_data().set_visited();
          continue;
        }

        if (node_data.is_connected()) {
          omp_set_lock(&lck);
          wire_num += searchEndNode(node_matrix[row][col], grid);
          omp_unset_lock(&lck);
        }
      }

      if (row % 1000 == 0) {
        LOG_INFO << "Patch layer " << layer_id << " Read rows : " << row << " / " << grid.get_info().node_row_num;
      }
    }
    LOG_INFO << "Patch layer " << layer_id << " Read rows : " << grid.get_info().node_row_num << " / " << grid.get_info().node_row_num;
  }

  omp_destroy_lock(&lck);
  return wire_num;
}

int LmLayoutDataManager::searchEndNode(LmNode& node_connected, LmLayerGrid& grid)
{
  int number = 0;

  for (LmNodeDirection direction_enum :
       {LmNodeDirection::lm_left, LmNodeDirection::lm_right, LmNodeDirection::lm_down, LmNodeDirection::lm_up}) {
    if (node_connected.get_node_data().is_direction(direction_enum)) {
      number += search_node_in_direction(node_connected, direction_enum, grid);
    }
  }

  return number;
}

LmNodeDirection LmLayoutDataManager::get_corner_orthogonal_direction(LmNode* node, LmNodeDirection direction)
{
  //   if (false == node->is_corner()) {
  //     return LmNodeDirection::kNone;
  //   }

  LmNodeData& node_data = node->get_node_data();
  if (direction == LmNodeDirection::lm_left || direction == LmNodeDirection::lm_right) {
    if (node_data.is_direction(LmNodeDirection::lm_down)) {
      return LmNodeDirection::lm_down;
    }

    if (node_data.is_direction(LmNodeDirection::lm_up)) {
      return LmNodeDirection::lm_up;
    }
  }

  if (direction == LmNodeDirection::lm_down || direction == LmNodeDirection::lm_up) {
    if (node_data.is_direction(LmNodeDirection::lm_left)) {
      return LmNodeDirection::lm_left;
    }

    if (node_data.is_direction(LmNodeDirection::lm_right)) {
      return LmNodeDirection::lm_right;
    }
  }

  return LmNodeDirection::kNone;
}

LmNodeDirection LmLayoutDataManager::get_opposite_direction(LmNodeDirection direction)
{
  LmNodeDirection opposite_direction = LmNodeDirection::kNone;
  switch (direction) {
    case LmNodeDirection::lm_left:
      opposite_direction = LmNodeDirection::lm_right;
      break;
    case LmNodeDirection::lm_right:
      opposite_direction = LmNodeDirection::lm_left;
      break;
    case LmNodeDirection::lm_up:
      opposite_direction = LmNodeDirection::lm_down;
      break;
    case LmNodeDirection::lm_down:
      opposite_direction = LmNodeDirection::lm_up;
      break;

    default:
      break;
  }

  return opposite_direction;
}

int LmLayoutDataManager::search_node_in_direction(LmNode& node_connected, LmNodeDirection direction, LmLayerGrid& grid)
{
  if (node_connected.get_node_data().is_direction_visited(direction)) {
    return 0;
  }

  int number = 0;

  LmNetWire wire;
  wire.set_start(&node_connected);

  auto* node_start = &node_connected;
  auto* node_end = travel_grid(node_start, direction, grid);
  while (node_end != nullptr) {
    if (node_end->get_node_data().is_connected()) {
      wire.set_end(node_end);
      wire.add_path(node_start, node_end);
      add_net_wire(node_start->get_node_data().get_net_id(), wire);
      number++;
      break;
    } else if (node_end->get_node_data().is_connecting()) {
      /// connecting means corner node with only two direction in this routing layer
      wire.add_path(node_start, node_end);
      number++;

      /// go to corner direciton node
      auto orthogonal_direction = get_corner_orthogonal_direction(node_end, direction);
      if (orthogonal_direction == LmNodeDirection::kNone) {
        // wire.set_end(node_end);
        // wire.add_path(node_start, node_end);
        // add_net_wire(node_start->get_node_data().get_net_id(), wire);
        // number++;
        LOG_INFO << "node_start [ " << node_start->get_row_id() << " , " << node_start->get_col_id() << " ]";
        LOG_INFO << "node_end [ " << node_end->get_row_id() << " , " << node_end->get_col_id() << " ]";
        break;
      }

      node_start = node_end;
      direction = orthogonal_direction;

      /// go to next node
      node_end = travel_grid(node_start, orthogonal_direction, grid);
    } else {
      /// wire not connected ?
      wire.set_end(node_end);
      wire.add_path(node_start, node_end);
      add_net_wire(node_start->get_node_data().get_net_id(), wire);
      number++;
      break;
    }
  }

  return number;
}

/// travel to next connecting or connected point
LmNode* LmLayoutDataManager::travel_grid(LmNode* node_start, LmNodeDirection direction, LmLayerGrid& grid)
{
  /// set direction visited states
  node_start->get_node_data().set_direction_visited(direction);

  auto& node_data = node_start->get_node_data();

  int row_travel = node_start->get_row_id();
  int col_travel = node_start->get_col_id();
  int row_delta = 0;
  int col_delta = 0;
  {
    switch (direction) {
      case LmNodeDirection::lm_left:
        --col_travel;
        col_delta = -1;
        break;
      case LmNodeDirection::lm_right:
        ++col_travel;
        col_delta = 1;
        break;
      case LmNodeDirection::lm_down:
        --row_travel;
        row_delta = -1;
        break;
      case LmNodeDirection::lm_up:
        ++row_travel;
        row_delta = 1;
        break;

      default:
        break;
    }
  }

  LmNode* node_end = nullptr;
  auto& node_matrix = grid.get_node_matrix();

  auto direction_opposite = get_opposite_direction(direction);
  while (false == grid.is_out_of_range(row_travel, col_travel)
         && false == node_matrix[row_travel][col_travel].get_node_data().is_direction_visited(direction_opposite)) {
    auto& travel_data = node_matrix[row_travel][col_travel].get_node_data();
    /// set visited
    travel_data.set_direction_visited(direction_opposite);
    if (travel_data.is_connecting() || travel_data.is_connected()) {
      /// success
      node_end = &node_matrix[row_travel][col_travel];
      break;
    }

    if (travel_data.is_net() == false || travel_data.get_net_id() != node_data.get_net_id()) {
      /// stop and return some error and return last node end
      break;
    }

    /// store this node as connect node
    node_end = &node_matrix[row_travel][col_travel];

    /// travel to next node
    row_travel += row_delta;
    col_travel += col_delta;
  }

  return node_end;
}

void LmLayoutDataManager::buildPatchs()
{
}

}  // namespace ilm