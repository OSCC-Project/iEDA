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

#include "CutEOLSpacingRule.hpp"
#include "DRCHeader.hpp"
#include "DifferentLayerCutSpacingRule.hpp"
#include "EnclosureEdgeRule.hpp"
#include "EnclosureParallelRule.hpp"
#include "PlanarRect.hpp"
#include "SameLayerCutSpacingRule.hpp"

namespace idrc {

class CutLayer
{
 public:
  CutLayer() = default;
  ~CutLayer() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  CutEOLSpacingRule& get_cut_eol_spacing_rule() { return _cut_eol_spacing_rule; }
  DifferentLayerCutSpacingRule& get_different_layer_cut_spacing_rule() { return _different_layer_cut_spacing_rule; }
  std::vector<EnclosureEdgeRule>& get_enclosure_edge_rule_list() { return _enclosure_edge_rule_list; }
  SameLayerCutSpacingRule& get_same_layer_cut_spacing_rule() { return _same_layer_cut_spacing_rule; }
  EnclosureParallelRule& get_enclosure_parallel_rule() { return _enclosure_parallel_rule; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  // function
 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  CutEOLSpacingRule _cut_eol_spacing_rule;
  DifferentLayerCutSpacingRule _different_layer_cut_spacing_rule;
  std::vector<EnclosureEdgeRule> _enclosure_edge_rule_list;
  SameLayerCutSpacingRule _same_layer_cut_spacing_rule;
  EnclosureParallelRule _enclosure_parallel_rule;
};

}  // namespace idrc
