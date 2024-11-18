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

  auto net_map = buildNetWires(true);
  if (net_map.size() > 0) {
    /// save to path
    LmLayoutFileIO file_io;
    return file_io.saveJson(path, net_map);
  }

  return false;
}

std::map<int, LmNet> LmLayoutDataManager::getGraph(std::string path)
{
  init();

  auto net_map = buildNetWires(true);
  if (net_map.size() > 0) {
    /// save to path
    LmLayoutFileIO file_io;
    file_io.saveJson(path, net_map);
  }

  return net_map;
}

void LmLayoutDataManager::init()
{
  LmLayoutInit layout_init(&_layout);
  layout_init.init();
}

void LmLayoutDataManager::add_net_wire(std::map<int, LmNet>& net_map, int net_id, LmNetWire wire)
{
  auto it = net_map.find(net_id);
  if (it != net_map.end()) {
    it->second.addWire(wire);
  } else {
    LmNet lm_net(net_id);
    lm_net.addWire(wire);
    auto result = net_map.insert(std::make_pair(net_id, lm_net));
  }
}

std::map<int, LmNet> LmLayoutDataManager::buildNetWires(bool b_graph)
{
  ieda::Stats stats;

  LOG_INFO << "LM build net wires start...";

  std::map<int, LmNet> net_map;

  int wire_num = 0;

  auto& patch_layers = _layout.get_patch_layers();
  for (auto& [layer_id, patch_layer] : patch_layers.get_patch_layer_map()) {
    if (patch_layer.is_routing()) {
      /// metal
      wire_num += buildRoutingLayer(layer_id, patch_layer, net_map);
    } else {
      /// via
      wire_num += buildCutLayer(layer_id, patch_layer, net_map);
    }
  }

  LOG_INFO << "Net number : " << net_map.size();
  LOG_INFO << "Wire number : " << wire_num;

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM build net wires end...";

  return net_map;
}

void LmLayoutDataManager::buildSteinerWire(LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map,
                                           std::vector<std::vector<bool>>& visited_matrix)
{
}

int LmLayoutDataManager::buildCutLayer(int layer_id, LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map)
{
  int wire_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& grid = patch_layer.get_grid();
  auto& node_matrix = grid.get_node_matrix();

  std::vector<std::vector<bool>> visited_matrix(grid.get_info().node_row_num, std::vector<bool>(grid.get_info().node_col_num, false));

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
          visited_matrix[row][col] = true;
          continue;
        }

        int net_id = node_data.get_net_id();
        /// skip node if node is not net
        if (net_id < 0) {
          visited_matrix[row][col] = true;
          continue;
        }

        omp_set_lock(&lck);
        LmNetWire wire(&node_matrix_bottom[row][col], &node_matrix_top[row][col]);
        wire.add_path(&node_matrix_bottom[row][col], &node_matrix_top[row][col]);
        add_net_wire(net_map, net_id, wire);
        ++wire_num;
        omp_unset_lock(&lck);

        visited_matrix[row][col] = true;
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

int LmLayoutDataManager::buildRoutingLayer(int layer_id, LmPatchLayer& patch_layer, std::map<int, LmNet>& net_map)
{
  int wire_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& grid = patch_layer.get_grid();
  auto& node_matrix = grid.get_node_matrix();

  std::vector<std::vector<bool>> visited_matrix(grid.get_info().node_row_num, std::vector<bool>(grid.get_info().node_col_num, false));

  if (true == patch_layer.is_routing()) {
    for (int row = 0; row < grid.get_info().node_row_num; ++row) {
      for (int col = 0; col < grid.get_info().node_col_num; ++col) {
        if (visited_matrix[row][col] == true) {
          /// skip visited node
          continue;
        }

        auto& node_data = node_matrix[row][col].get_node_data();
        int net_id = node_data.get_net_id();
        /// skip node if node is not net
        if (net_id < 0) {
          visited_matrix[row][col] = true;
          continue;
        }

        if (node_data.get_status() == LmNodeStatus::lm_connected) {
          omp_set_lock(&lck);
          wire_num += searchEndNode(node_matrix[row][col], grid, net_map, visited_matrix);
          omp_unset_lock(&lck);
        }

        visited_matrix[row][col] = true;
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

int LmLayoutDataManager::searchEndNode(LmNode& node_connected, LmLayerGrid& grid, std::map<int, LmNet>& net_map,
                                       std::vector<std::vector<bool>>& visited_matrix)
{
  int number = 0;

  for (LmNodeDirection direction_enum :
       {LmNodeDirection::lm_left, LmNodeDirection::lm_right, LmNodeDirection::lm_down, LmNodeDirection::lm_up}) {
    if (node_connected.get_node_data().is_direction(direction_enum)) {
      number += search_node_in_direction(node_connected, direction_enum, grid, net_map, visited_matrix);
    }
  }

  return number;
}

int LmLayoutDataManager::search_node_in_direction(LmNode& node_connected, LmNodeDirection direction, LmLayerGrid& grid,
                                                  std::map<int, LmNet>& net_map, std::vector<std::vector<bool>>& visited_matrix)
{
  auto get_corner_orthogonal_direction = [](LmNode* node, LmNodeDirection direction) -> LmNodeDirection {
    if (false == node->is_corner()) {
      return LmNodeDirection::kNone;
    }

    LmNodeData& node_data = node->get_node_data();
    if (direction == LmNodeDirection::lm_left || direction == LmNodeDirection::lm_right) {
      if (node_data.is_direction(LmNodeDirection::lm_down) || node_data.is_direction(LmNodeDirection::lm_middle)) {
        return LmNodeDirection::lm_down;
      }

      if (node_data.is_direction(LmNodeDirection::lm_up) || node_data.is_direction(LmNodeDirection::lm_middle)) {
        return LmNodeDirection::lm_up;
      }
    }

    if (direction == LmNodeDirection::lm_down || direction == LmNodeDirection::lm_up) {
      if (node_data.is_direction(LmNodeDirection::lm_left) || node_data.is_direction(LmNodeDirection::lm_middle)) {
        return LmNodeDirection::lm_left;
      }

      if (node_data.is_direction(LmNodeDirection::lm_right) || node_data.is_direction(LmNodeDirection::lm_middle)) {
        return LmNodeDirection::lm_right;
      }
    }

    return LmNodeDirection::kNone;
  };

  int number = 0;

  auto* node_start = &node_connected;
  LmNetWire wire(node_start);
  auto* node_search = travel_grid(node_start, direction, grid, visited_matrix);
  while (node_search != nullptr) {
    if (node_search->get_node_data().get_status() == LmNodeStatus::lm_connected) {
      wire.set_end(node_search);
      wire.add_path(node_start, node_search);
      add_net_wire(net_map, node_start->get_node_data().get_net_id(), wire);
      number++;
      break;
    } else if (node_search->get_node_data().get_status() == LmNodeStatus::lm_connecting) {
      /// connecting means corner node with only two direction in this routing layer
      wire.add_path(node_start, node_search);
      number++;

      /// go to corner direciton node
      auto orthogonal_direction = get_corner_orthogonal_direction(node_search, direction);
      if (orthogonal_direction == LmNodeDirection::kNone) {
        wire.set_end(node_search);
        add_net_wire(net_map, node_start->get_node_data().get_net_id(), wire);
        break;
      }
      node_start = node_search;

      /// go to next node
      node_search = travel_grid(node_start, orthogonal_direction, grid, visited_matrix);
    } else {
      /// wire not connected ?
      wire.set_end(node_search);
      wire.add_path(node_start, node_search);
      add_net_wire(net_map, node_start->get_node_data().get_net_id(), wire);
      number++;
      break;
    }
  }

  return number;
}

/// travel to next connecting or connected point
LmNode* LmLayoutDataManager::travel_grid(LmNode* node_start, LmNodeDirection direction, LmLayerGrid& grid,
                                         std::vector<std::vector<bool>>& visited_matrix)
{
  LmNode* node_connect = nullptr;
  auto& node_matrix = grid.get_node_matrix();

  int row_travel = node_start->get_row_id();
  int col_travel = node_start->get_col_id();
  auto& node_data = node_start->get_node_data();

  int row_delta = 0;
  int col_delta = 0;
  if (node_data.is_direction(direction) && direction == LmNodeDirection::lm_left) {
    --col_travel;
    col_delta = -1;
  }

  if (node_data.is_direction(direction) && direction == LmNodeDirection::lm_right) {
    ++col_travel;
    col_delta = 1;
  }

  if (node_data.is_direction(direction) && direction == LmNodeDirection::lm_down) {
    --row_travel;
    row_delta = -1;
  }

  if (node_data.is_direction(direction) && direction == LmNodeDirection::lm_up) {
    ++row_travel;
    row_delta = 1;
  }

  while (false == grid.is_out_of_range(row_travel, col_travel) && false == visited_matrix[row_travel][col_travel]) {
    auto* this_node = &node_matrix[row_travel][col_travel];
    visited_matrix[row_travel][col_travel] = true;

    auto& travel_data = this_node->get_node_data();
    if (travel_data.get_status() == LmNodeStatus::lm_connecting || travel_data.get_status() == LmNodeStatus::lm_connected
        || travel_data.is_net() == false || travel_data.get_net_id() != node_data.get_net_id()) {
      /// find node need to be record
      //   if (true == this_node->is_steiner_point() || true == this_node->is_corner()) {
      //     visited_matrix[row_travel][col_travel] = false;
      //   }
      if (travel_data.get_status() == LmNodeStatus::lm_connecting || travel_data.get_status() == LmNodeStatus::lm_connected) {
        visited_matrix[row_travel][col_travel] = false;
      }
      return node_connect;
    }

    /// store this node as connect node
    node_connect = this_node;

    /// travel to next node
    row_travel += row_delta;
    col_travel += col_delta;
  }

  return node_connect;
}

void LmLayoutDataManager::buildPatchs()
{
}

}  // namespace ilm