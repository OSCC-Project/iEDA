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

#include "PlanarRect.hpp"
#include "RTHeader.hpp"

namespace irt {

class CutLayer
{
 public:
  CutLayer() = default;
  ~CutLayer() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  int32_t get_curr_prl_spacing() const { return _curr_prl_spacing; }
  int32_t get_curr_x_spacing() const { return _curr_x_spacing; }
  int32_t get_curr_y_spacing() const { return _curr_y_spacing; }
  int32_t get_curr_eol_x_spacing() const { return _curr_eol_x_spacing; }
  int32_t get_curr_eol_y_spacing() const { return _curr_eol_y_spacing; }
  int32_t get_above_prl_spacing() const { return _above_prl_spacing; }
  int32_t get_above_x_spacing() const { return _above_x_spacing; }
  int32_t get_above_y_spacing() const { return _above_y_spacing; }
  int32_t get_below_prl_spacing() const { return _below_prl_spacing; }
  int32_t get_below_x_spacing() const { return _below_x_spacing; }
  int32_t get_below_y_spacing() const { return _below_y_spacing; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_curr_prl_spacing(const int32_t curr_prl_spacing) { _curr_prl_spacing = curr_prl_spacing; }
  void set_curr_x_spacing(const int32_t curr_x_spacing) { _curr_x_spacing = curr_x_spacing; }
  void set_curr_y_spacing(const int32_t curr_y_spacing) { _curr_y_spacing = curr_y_spacing; }
  void set_curr_eol_x_spacing(const int32_t curr_eol_x_spacing) { _curr_eol_x_spacing = curr_eol_x_spacing; }
  void set_curr_eol_y_spacing(const int32_t curr_eol_y_spacing) { _curr_eol_y_spacing = curr_eol_y_spacing; }
  void set_above_prl_spacing(const int32_t above_prl_spacing) { _above_prl_spacing = above_prl_spacing; }
  void set_above_x_spacing(const int32_t above_x_spacing) { _above_x_spacing = above_x_spacing; }
  void set_above_y_spacing(const int32_t above_y_spacing) { _above_y_spacing = above_y_spacing; }
  void set_below_prl_spacing(const int32_t below_prl_spacing) { _below_prl_spacing = below_prl_spacing; }
  void set_below_x_spacing(const int32_t below_x_spacing) { _below_x_spacing = below_x_spacing; }
  void set_below_y_spacing(const int32_t below_y_spacing) { _below_y_spacing = below_y_spacing; }
  // function
 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  // curr spacing
  int32_t _curr_prl_spacing = -1;
  int32_t _curr_x_spacing = -1;
  int32_t _curr_y_spacing = -1;
  // curr eol
  int32_t _curr_eol_x_spacing = -1;
  int32_t _curr_eol_y_spacing = -1;
  // above spacing
  int32_t _above_prl_spacing = -1;
  int32_t _above_x_spacing = -1;
  int32_t _above_y_spacing = -1;
  // below spacing
  int32_t _below_prl_spacing = -1;
  int32_t _below_x_spacing = -1;
  int32_t _below_y_spacing = -1;
};

}  // namespace irt
