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

#include "Direction.hpp"
#include "Logger.hpp"
#include "Orientation.hpp"
#include "PlanarRect.hpp"
#include "ScaleAxis.hpp"
#include "SpacingTable.hpp"

namespace irt {

class RoutingLayer
{
 public:
  RoutingLayer() = default;
  ~RoutingLayer() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  irt_int get_min_width() const { return _min_width; }
  irt_int get_min_area() const { return _min_area; }
  Direction& get_prefer_direction() { return _prefer_direction; }
  ScaleAxis& get_track_axis() { return _track_axis; }
  SpacingTable& get_spacing_table() { return _spacing_table; }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const irt_int layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_min_width(const irt_int min_width) { _min_width = min_width; }
  void set_min_area(const irt_int min_area) { _min_area = min_area; }
  void set_prefer_direction(const Direction& prefer_direction) { _prefer_direction = prefer_direction; }
  void set_track_axis(const ScaleAxis& track_axis) { _track_axis = track_axis; }
  void set_spacing_table(const SpacingTable& spacing_table) { _spacing_table = spacing_table; }

  // function
  bool isPreferH() const { return _prefer_direction == Direction::kHorizontal; }
  std::vector<ScaleGrid>& getXTrackGridList() { return _track_axis.get_x_grid_list(); }
  std::vector<ScaleGrid>& getYTrackGridList() { return _track_axis.get_y_grid_list(); }
  std::vector<ScaleGrid>& getPreferTrackGridList() { return isPreferH() ? getYTrackGridList() : getXTrackGridList(); }
  std::vector<ScaleGrid>& getNonpreferTrackGridList() { return isPreferH() ? getXTrackGridList() : getYTrackGridList(); }
  irt_int getMinSpacing(const PlanarRect& rect)
  {
    std::vector<irt_int>& width_list = _spacing_table.get_width_list();
    GridMap<irt_int>& width_parallel_length_map = _spacing_table.get_width_parallel_length_map();

    for (size_t i = 0; (i + 1) < width_list.size(); i++) {
      if (width_list[i] <= rect.getWidth() && rect.getWidth() < width_list[i + 1]) {
        return width_parallel_length_map[i][0];
      }
    }
    return width_parallel_length_map[width_parallel_length_map.get_x_size() - 1][0];
  }
  std::vector<Orientation> getPreferOrientationList()
  {
    std::vector<Orientation> orientation_list;
    if (_prefer_direction == Direction::kHorizontal) {
      orientation_list = {Orientation::kEast, Orientation::kWest};
    } else {
      orientation_list = {Orientation::kSouth, Orientation::kNorth};
    }
    return orientation_list;
  }

 private:
  irt_int _layer_idx = -1;
  irt_int _layer_order = -1;
  std::string _layer_name;
  irt_int _min_width = 0;
  irt_int _min_area = 0;
  Direction _prefer_direction = Direction::kNone;
  ScaleAxis _track_axis;
  SpacingTable _spacing_table;
};

}  // namespace irt
