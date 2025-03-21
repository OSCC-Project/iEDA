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
  int32_t get_pitch() const { return _pitch; }
  int32_t get_min_width() const { return _min_width; }
  int32_t get_min_area() const { return _min_area; }
  int32_t get_min_hole() const { return _min_hole; }
  SpacingTable& get_prl_spacing_table() { return _prl_spacing_table; }
  int32_t get_eol_spacing() const { return _eol_spacing; }
  int32_t get_eol_ete() const { return _eol_ete; }
  int32_t get_eol_within() const { return _eol_within; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_pitch(const int32_t pitch) { _pitch = pitch; }
  void set_min_width(const int32_t min_width) { _min_width = min_width; }
  void set_min_area(const int32_t min_area) { _min_area = min_area; }
  void set_min_hole(const int32_t min_hole) { _min_hole = min_hole; }
  void set_prl_spacing_table(const SpacingTable& prl_spacing_table) { _prl_spacing_table = prl_spacing_table; }
  void set_eol_spacing(const int32_t eol_spacing) { _eol_spacing = eol_spacing; }
  void set_eol_ete(const int32_t eol_ete) { _eol_ete = eol_ete; }
  void set_eol_within(const int32_t eol_within) { _eol_within = eol_within; }
  // function

 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  int32_t _pitch = 0;
  // min width
  int32_t _min_width = 0;
  // min area
  int32_t _min_area = 0;
  // min hole
  int32_t _min_hole = 0;
  // prl
  SpacingTable _prl_spacing_table;
  // eol
  int32_t _eol_spacing = -1;
  int32_t _eol_ete = -1;
  int32_t _eol_within = -1;
};

}  // namespace idrc
