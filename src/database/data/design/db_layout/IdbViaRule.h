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
/**
 * @project		iDB
 * @file		IdbViaRule.h
 * @date		25/05/2021
 * @version		0.1
 * @description

        Describe Via Rule & Via Rule Generate information in deflefref P.148 & P.150:
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "IdbLayer.h"

namespace idb {

using std::string;
using std::vector;

/*Via rule generate in P.150
Via Rule Generate
VIARULE viaRuleName GENERATE [DEFAULT]
  LAYER routingLayerName ;
    ENCLOSURE overhang1 overhang2 ;
    [WIDTH minWidth TO maxWidth ;]
  LAYER routingLayerName ;
    ENCLOSURE overhang1 overhang2 ;
    [WIDTH minWidth TO maxWidth ;]
  LAYER cutLayerName ;
    RECT pt pt ;
    SPACING xSpacing BY ySpacing ;
    [RESISTANCE resistancePerCut ;]
END viaRuleName
*/
class IdbViaRuleGenerate
{
 public:
  IdbViaRuleGenerate();
  ~IdbViaRuleGenerate();

  // getter
  const string& get_name() const { return _name; }
  IdbLayerRouting* get_layer_bottom() { return _layer_bottom; }
  bool is_bottom_layer_set() { return _layer_bottom != nullptr ? true : false; }
  IdbLayerCutEnclosure* get_enclosure_bottom() { return _enclosure_bottom; }
  IdbLayerRouting* get_layer_top() { return _layer_top; }
  bool is_top_layer_set() { return _layer_top != nullptr ? true : false; }
  IdbLayerCutEnclosure* get_enclosure_top() { return _enclosure_top; }
  IdbLayerCut* get_layer_cut() { return _layer_cut; }
  IdbRect* get_cut_rect() { return _cut_rect; }
  const int32_t get_spacing_x() const { return _spacing_x; }
  const int32_t get_spacing_y() const { return _spacing_y; }

  // setter
  void set_name(string name) { _name = name; }
  void set_layer_bottom(IdbLayerRouting* layer_bottom) { _layer_bottom = layer_bottom; }
  void set_enclosure_bottom(IdbLayerCutEnclosure* enclosure) { _enclosure_bottom = enclosure; }
  void set_layer_top(IdbLayerRouting* layer_top) { _layer_top = layer_top; }
  void set_enclosure_top(IdbLayerCutEnclosure* enclosure) { _enclosure_top = enclosure; }
  void set_layer_cut(IdbLayerCut* layer_cut) { _layer_cut = layer_cut; }
  void set_cut_rect(IdbRect* rect) { _cut_rect = rect; }
  void set_cut_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void set_spacing(int32_t x, int32_t y)
  {
    _spacing_x = x;
    _spacing_y = y;
  }

  // operator
  void swap_routing_layer();

 private:
  string _name;
  IdbLayerRouting* _layer_bottom;
  IdbLayerCutEnclosure* _enclosure_bottom;
  IdbLayerCut* _layer_cut;
  IdbRect* _cut_rect;
  int32_t _spacing_x;
  int32_t _spacing_y;
  IdbLayerRouting* _layer_top;
  IdbLayerCutEnclosure* _enclosure_top;
};
/*
You should only use VIARULE GENERATE statements to create a via for the
intersection of two special wires. In earlier versions of LEF, VIARULE GENERATE was not
complete enough to cover all situations. In those cases, a fixed VIARULE (without a
GENERATE keyword) was sometimes used. This is no longer required.
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbRule
{
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbViaRuleList
{
 public:
  IdbViaRuleList();
  ~IdbViaRuleList();

  // getter
  vector<IdbViaRuleGenerate*>& get_rule_list() { return _via_rule_generate_list; }
  const int32_t get_num_via_rule_generate() const { return _num_rule_generate; }
  IdbViaRuleGenerate* find_via_rule_generate(string name);
  IdbViaRuleGenerate* find_via_rule_generate(int32_t index);

  // setter

  void set_num_via_rule(uint32_t number) { _num_rule_generate = number; }
  IdbViaRuleGenerate* add_via_rule_generate(IdbViaRuleGenerate* via_rule = nullptr);
  IdbViaRuleGenerate* add_via_rule_generate(string name);

  void reset();

  // operator
  void init_via_rule_list(int32_t size) { _via_rule_generate_list.reserve(size); }

 private:
  int32_t _num_rule_generate;
  vector<IdbViaRuleGenerate*> _via_rule_generate_list;
};

}  // namespace idb
