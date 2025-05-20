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
#include <optional>

#include "Logger.hpp"
#include "PlanarRect.hpp"
#include "SpacingTable.hpp"

namespace idrc {

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
  int32_t get_pitch() const { return _pitch; }
  int32_t get_min_width() const { return _min_width; }
  int32_t get_max_width() const { return _max_width; }
  int32_t get_min_area() const { return _min_area; }
  int32_t get_min_hole_area() const { return _min_hole_area; }
  int32_t get_min_step() const { return _min_step; }
  int32_t get_max_edges() const { return _max_edges; }
  int32_t get_lef58_min_step() const { return _lef58_min_step; }
  int32_t get_lef58_min_adjacent_length() const { return _lef58_min_adjacent_length; }
  int32_t get_notch_spacing() const { return _notch_spacing; }
  int32_t get_notch_length() const { return _notch_length; }
  std::optional<int32_t> get_concave_ends() const { return _concave_ends; }
  SpacingTable& get_prl_spacing_table() { return _prl_spacing_table; }
  int32_t get_eol_spacing() const { return _eol_spacing; }
  int32_t get_eol_ete() const { return _eol_ete; }
  int32_t get_eol_within() const { return _eol_within; }
  bool get_has_corner_fill() const { return _has_corner_fill; }
  int32_t get_corner_fill_spacing() const { return _corner_fill_spacing; }
  int32_t get_edge_length_1() const { return _edge_length_1; }
  int32_t get_edge_length_2() const { return _edge_length_2; }
  int32_t get_adjacent_eol() const { return _adjacent_eol; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_prefer_direction(const Direction& prefer_direction) { _prefer_direction = prefer_direction; }
  void set_pitch(const int32_t pitch) { _pitch = pitch; }
  void set_min_width(const int32_t min_width) { _min_width = min_width; }
  void set_max_width(const int32_t max_width) { _max_width = max_width; }
  void set_min_area(const int32_t min_area) { _min_area = min_area; }
  void set_min_hole_area(const int32_t min_hole_area) { _min_hole_area = min_hole_area; }
  void set_min_step(const int32_t min_step) { _min_step = min_step; }
  void set_max_edges(const int32_t max_edges) { _max_edges = max_edges; }
  void set_lef58_min_step(const int32_t lef58_min_step) { _lef58_min_step = lef58_min_step; }
  void set_lef58_min_adjacent_length(const int32_t lef58_min_adjacent_length) { _lef58_min_adjacent_length = lef58_min_adjacent_length; }
  void set_notch_spacing(const int32_t notch_spacing) { _notch_spacing = notch_spacing; }
  void set_notch_length(const int32_t notch_length) { _notch_length = notch_length; }
  void set_concave_ends(const std::optional<int32_t> concave_ends) { _concave_ends = concave_ends; }
  void set_prl_spacing_table(const SpacingTable& prl_spacing_table) { _prl_spacing_table = prl_spacing_table; }
  void set_eol_spacing(const int32_t eol_spacing) { _eol_spacing = eol_spacing; }
  void set_eol_ete(const int32_t eol_ete) { _eol_ete = eol_ete; }
  void set_eol_within(const int32_t eol_within) { _eol_within = eol_within; }
  void set_has_corner_fill(const bool has_corner_fill) { _has_corner_fill = has_corner_fill; }
  void set_corner_fill_spacing(const int32_t corner_fill_spacing) { _corner_fill_spacing = corner_fill_spacing; }
  void set_edge_length_1(const int32_t edge_length_1) { _edge_length_1 = edge_length_1; }
  void set_edge_length_2(const int32_t edge_length_2) { _edge_length_2 = edge_length_2; }
  void set_adjacent_eol(const int32_t adjacent_eol) { _adjacent_eol = adjacent_eol; }
  // function
  bool isPreferH() const { return _prefer_direction == Direction::kHorizontal; }
  int32_t getMaxSpacing() { return _prl_spacing_table.get_width_parallel_length_map().back().back(); }
  int32_t getSpacing(int32_t width, int32_t parallel_length)
  {
    std::vector<int32_t>& width_list = _prl_spacing_table.get_width_list();
    std::vector<int32_t>& parallel_length_list = _prl_spacing_table.get_parallel_length_list();
    GridMap<int32_t>& width_parallel_length_map = _prl_spacing_table.get_width_parallel_length_map();

    int32_t width_idx = static_cast<int32_t>(width_list.size()) - 1;
    for (int32_t i = width_idx; 0 <= i; i--) {
      if (width_list[i] <= width) {
        width_idx = i;
        break;
      }
    }
    int32_t parallel_length_idx = static_cast<int32_t>(parallel_length_list.size()) - 1;
    for (int32_t i = parallel_length_idx; 0 <= i; i--) {
      if (parallel_length_list[i] <= parallel_length) {
        parallel_length_idx = i;
        break;
      }
    }
    return width_parallel_length_map[width_idx][parallel_length_idx];
  }

 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  Direction _prefer_direction = Direction::kNone;
  int32_t _pitch = -1;
  // min width
  int32_t _min_width = -1;
  // max width
  int32_t _max_width = -1;
  // min area
  int32_t _min_area = -1;
  // min hole
  int32_t _min_hole_area = -1;
  // min step
  int32_t _min_step = -1;
  int32_t _max_edges = -1;
  int32_t _lef58_min_step = -1;
  int32_t _lef58_min_adjacent_length = -1;
  // notch
  int32_t _notch_spacing = -1;
  int32_t _notch_length = -1;
  std::optional<int32_t> _concave_ends;
  // prl
  SpacingTable _prl_spacing_table;
  // eol
  int32_t _eol_spacing = -1;
  int32_t _eol_ete = -1;
  int32_t _eol_within = -1;
  // corner fill
  bool _has_corner_fill = false;
  int32_t _corner_fill_spacing = -1;
  int32_t _edge_length_1 = -1;
  int32_t _edge_length_2 = -1;
  int32_t _adjacent_eol = -1;
};

}  // namespace idrc
