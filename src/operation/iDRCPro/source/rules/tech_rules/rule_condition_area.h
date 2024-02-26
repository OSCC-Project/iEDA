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

#include "IdbRoutingLayerLef58Property.h"
#include "rule_basic.h"
#include "rule_condition_map.h"

namespace idrc {

class ConditionRuleArea : public ConditionRule
{
 public:
  ConditionRuleArea(RuleType type, int min_area) : ConditionRule(type, min_area) {}
  ~ConditionRuleArea() {}

  int get_min_area() { return get_value(); }

 private:
};

class ConditionRuleAreaLef58 : public ConditionRule
{
 public:
  ConditionRuleAreaLef58(RuleType type, int min_area, idb::routinglayer::Lef58Area* area_rule)
      : ConditionRule(type, min_area), _area_rule(area_rule)
  {
  }
  ~ConditionRuleAreaLef58() {}

  idb::routinglayer::Lef58Area* get_rule() { return _area_rule; }

 private:
  idb::routinglayer::Lef58Area* _area_rule;
};

class RulesMapArea : public RulesConditionMap
{
 public:
  RulesMapArea(RuleType type) : RulesConditionMap(type) {}
  ~RulesMapArea() {}

 private:
};

}  // namespace idrc