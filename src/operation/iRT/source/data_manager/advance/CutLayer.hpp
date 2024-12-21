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
  int32_t get_curr_spacing() const { return _curr_spacing; }
  int32_t get_curr_prl() const { return _curr_prl; }
  int32_t get_curr_prl_spacing() const { return _curr_prl_spacing; }
  int32_t get_curr_eol_spacing() const { return _curr_eol_spacing; }
  int32_t get_curr_eol_prl() const { return _curr_eol_prl; }
  int32_t get_curr_eol_prl_spacing() const { return _curr_eol_prl_spacing; }
  int32_t get_above_spacing() const { return _above_spacing; }
  int32_t get_above_prl() const { return _above_prl; }
  int32_t get_above_prl_spacing() const { return _above_prl_spacing; }
  int32_t get_below_spacing() const { return _below_spacing; }
  int32_t get_below_prl() const { return _below_prl; }
  int32_t get_below_prl_spacing() const { return _below_prl_spacing; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_curr_spacing(const int32_t curr_spacing) { _curr_spacing = curr_spacing; }
  void set_curr_prl(const int32_t curr_prl) { _curr_prl = curr_prl; }
  void set_curr_prl_spacing(const int32_t curr_prl_spacing) { _curr_prl_spacing = curr_prl_spacing; }
  void set_curr_eol_spacing(const int32_t curr_eol_spacing) { _curr_eol_spacing = curr_eol_spacing; }
  void set_curr_eol_prl(const int32_t curr_eol_prl) { _curr_eol_prl = curr_eol_prl; }
  void set_curr_eol_prl_spacing(const int32_t curr_eol_prl_spacing) { _curr_eol_prl_spacing = curr_eol_prl_spacing; }
  void set_above_spacing(const int32_t above_spacing) { _above_spacing = above_spacing; }
  void set_above_prl(const int32_t above_prl) { _above_prl = above_prl; }
  void set_above_prl_spacing(const int32_t above_prl_spacing) { _above_prl_spacing = above_prl_spacing; }
  void set_below_spacing(const int32_t below_spacing) { _below_spacing = below_spacing; }
  void set_below_prl(const int32_t below_prl) { _below_prl = below_prl; }
  void set_below_prl_spacing(const int32_t below_prl_spacing) { _below_prl_spacing = below_prl_spacing; }
  // function
 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  // curr prl
  int32_t _curr_spacing = -1;
  int32_t _curr_prl = -1;
  int32_t _curr_prl_spacing = -1;
  // curr eol
  int32_t _curr_eol_spacing = -1;
  int32_t _curr_eol_prl = -1;
  int32_t _curr_eol_prl_spacing = -1;
  // above prl
  int32_t _above_spacing = -1;
  int32_t _above_prl = -1;
  int32_t _above_prl_spacing = -1;
  // below prl
  int32_t _below_spacing = -1;
  int32_t _below_prl = -1;
  int32_t _below_prl_spacing = -1;
};

}  // namespace irt
