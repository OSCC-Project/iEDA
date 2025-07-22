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
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  Direction& get_prefer_direction() { return _prefer_direction; }
  ScaleAxis& get_track_axis() { return _track_axis; }
  int32_t get_min_width() const { return _min_width; }
  int32_t get_min_area() const { return _min_area; }
  int32_t get_notch_spacing() const { return _notch_spacing; }
  SpacingTable& get_prl_spacing_table() { return _prl_spacing_table; }
  int32_t get_eol_spacing() const { return _eol_spacing; }
  int32_t get_eol_ete() const { return _eol_ete; }
  int32_t get_eol_within() const { return _eol_within; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_prefer_direction(const Direction& prefer_direction) { _prefer_direction = prefer_direction; }
  void set_track_axis(const ScaleAxis& track_axis) { _track_axis = track_axis; }
  void set_min_width(const int32_t min_width) { _min_width = min_width; }
  void set_min_area(const int32_t min_area) { _min_area = min_area; }
  void set_notch_spacing(const int32_t notch_spacing) { _notch_spacing = notch_spacing; }
  void set_prl_spacing_table(const SpacingTable& prl_spacing_table) { _prl_spacing_table = prl_spacing_table; }
  void set_eol_spacing(const int32_t eol_spacing) { _eol_spacing = eol_spacing; }
  void set_eol_ete(const int32_t eol_ete) { _eol_ete = eol_ete; }
  void set_eol_within(const int32_t eol_within) { _eol_within = eol_within; }
  // function
  bool isPreferH() const { return _prefer_direction == Direction::kHorizontal; }
  std::vector<ScaleGrid>& getXTrackGridList() { return _track_axis.get_x_grid_list(); }
  std::vector<ScaleGrid>& getYTrackGridList() { return _track_axis.get_y_grid_list(); }
  std::vector<ScaleGrid>& getPreferTrackGridList() { return isPreferH() ? getYTrackGridList() : getXTrackGridList(); }
  std::vector<ScaleGrid>& getNonPreferTrackGridList() { return isPreferH() ? getXTrackGridList() : getYTrackGridList(); }
  int32_t getPRLSpacing(const PlanarRect& rect) { return getPRLSpacing(rect.getWidth()); }
  int32_t getPRLSpacing(const int32_t width)
  {
    std::vector<int32_t>& width_list = _prl_spacing_table.get_width_list();
    GridMap<int32_t>& width_parallel_length_map = _prl_spacing_table.get_width_parallel_length_map();

    int32_t width_idx = static_cast<int32_t>(width_list.size()) - 1;
    for (int32_t i = width_idx; 0 <= i; i--) {
      if (width_list[i] <= width) {
        width_idx = i;
        break;
      }
    }
    return width_parallel_length_map[width_idx].back();
  }

 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  Direction _prefer_direction = Direction::kNone;
  ScaleAxis _track_axis;
  // min width
  int32_t _min_width = -1;
  // min area
  int32_t _min_area = -1;
  // notch
  int32_t _notch_spacing = -1;
  // prl
  SpacingTable _prl_spacing_table;
  // eol
  int32_t _eol_spacing = -1;
  int32_t _eol_ete = -1;
  int32_t _eol_within = -1;
};

}  // namespace irt
