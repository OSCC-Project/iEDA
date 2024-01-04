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

#include <map>
#include <set>

#include "IdbLayer.h"
#include "rule_basic.h"
#include "rule_condition_map.h"

namespace idrc {

class ConditionRuleSpacingRange : public ConditionRule
{
 public:
  ConditionRuleSpacingRange(RuleType type, int spacing, idb::IdbLayerSpacing* spacing_range)
      : ConditionRule(type, spacing), _spacing_range(spacing_range)
  {
  }
  ~ConditionRuleSpacingRange() {}

  idb::IdbLayerSpacing* get_spacing_range() { return _spacing_range; }

 private:
  idb::IdbLayerSpacing* _spacing_range;
};

class ConditionRuleSpacingPRL : public ConditionRule
{
 public:
  ConditionRuleSpacingPRL(RuleType type, int spacing) : ConditionRule(type, spacing) {}
  ~ConditionRuleSpacingPRL() {}

  void addSpacingItem(int width, int prl_length) { _spacing_items[width].insert(prl_length); }

  bool isMatchCondition(int width, int prl_length);

 private:
  std::map<int, std::set<int, std::greater<int>>, std::greater<int>> _spacing_items;  // map int : witdh; set int : parallel run length
};

class ConditionRuleJogToJog : public ConditionRule
{
 public:
  ConditionRuleJogToJog(RuleType type, int spacing, idb::routinglayer::Lef58SpacingTableJogToJog* jog_to_jog)
      : ConditionRule(type, spacing), _jog_to_jog(jog_to_jog)
  {
  }
  ~ConditionRuleJogToJog() {}

  idb::routinglayer::Lef58SpacingTableJogToJog* get_jog_to_jog() { return _jog_to_jog; }
  void addWidth(int width, idb::routinglayer::Lef58SpacingTableJogToJog::Width* jog_width_class) { _width_map[width] = jog_width_class; }
  std::vector<idb::routinglayer::Lef58SpacingTableJogToJog::Width*> findWidthRule(int wire_max_width);

 private:
  idb::routinglayer::Lef58SpacingTableJogToJog* _jog_to_jog;
  std::map<int, idb::routinglayer::Lef58SpacingTableJogToJog::Width*, std::greater<int>> _width_map;  // int : width
};

class RulesMapSpacing : public RulesConditionMap
{
 public:
  RulesMapSpacing(RuleType type) : RulesConditionMap(type) {}
  ~RulesMapSpacing() {}

 private:
};

}  // namespace idrc