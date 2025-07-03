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

#include "CornerFillSpacingRule.hpp"
#include "DRCHeader.hpp"
#include "EndOfLineSpacingRule.hpp"
#include "Logger.hpp"
#include "MaximumWidthRule.hpp"
#include "MinHoleRule.hpp"
#include "MinStepRule.hpp"
#include "MinimumAreaRule.hpp"
#include "MinimumCutRule.hpp"
#include "MinimumWidthRule.hpp"
#include "NonsufficientMetalOverlapRule.hpp"
#include "NotchSpacingRule.hpp"
#include "ParallelRunLengthSpacingRule.hpp"
#include "PlanarRect.hpp"

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
  CornerFillSpacingRule& get_corner_fill_spacing_rule() { return _corner_fill_spacing_rule; }
  std::vector<EndOfLineSpacingRule>& get_end_of_line_spacing_rule_list() { return _end_of_line_spacing_rule_list; }
  MaximumWidthRule& get_maximum_width_rule() { return _maximum_width_rule; }
  MinHoleRule& get_min_hole_rule() { return _min_hole_rule; }
  MinimumAreaRule& get_minimum_area_rule() { return _minimum_area_rule; }
  std::vector<MinimumCutRule>& get_minimum_cut_rule_list() { return _minimum_cut_rule_list; }
  MinimumWidthRule& get_minimum_width_rule() { return _minimum_width_rule; }
  MinStepRule& get_min_step_rule() { return _min_step_rule; }
  NonsufficientMetalOverlapRule& get_nonsufficient_metal_overlap_rule() { return _nonsufficient_metal_overlap_rule; }
  NotchSpacingRule& get_notch_spacing_rule() { return _notch_spacing_rule; }
  ParallelRunLengthSpacingRule& get_parallel_run_length_spacing_rule() { return _parallel_run_length_spacing_rule; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_prefer_direction(const Direction& prefer_direction) { _prefer_direction = prefer_direction; }
  void set_pitch(const int32_t pitch) { _pitch = pitch; }
  // function
  bool isPreferH() const { return _prefer_direction == Direction::kHorizontal; }

 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  Direction _prefer_direction = Direction::kNone;
  int32_t _pitch = -1;
  CornerFillSpacingRule _corner_fill_spacing_rule;
  std::vector<EndOfLineSpacingRule> _end_of_line_spacing_rule_list;
  MaximumWidthRule _maximum_width_rule;
  MinHoleRule _min_hole_rule;
  MinimumAreaRule _minimum_area_rule;
  std::vector<MinimumCutRule> _minimum_cut_rule_list;
  MinimumWidthRule _minimum_width_rule;
  MinStepRule _min_step_rule;
  NonsufficientMetalOverlapRule _nonsufficient_metal_overlap_rule;
  NotchSpacingRule _notch_spacing_rule;
  ParallelRunLengthSpacingRule _parallel_run_length_spacing_rule;
};

}  // namespace idrc
