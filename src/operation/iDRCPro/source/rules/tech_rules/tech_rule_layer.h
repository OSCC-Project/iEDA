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

#include "rule_enum.h"
#include "idm.h"
#include "rule_condition_area.h"
#include "rule_condition_map.h"

namespace idrc {

class ConditionRuleLayer
{
 public:
  ConditionRuleLayer() {}
  ~ConditionRuleLayer() {}

  RulesConditionMap* get_condition_map(RuleType type) { return _rule_condition_map[type]; }
  void set_condition(RuleType type, RulesConditionMap* condition_map) { _rule_condition_map[type] = condition_map; }

 private:
  std::map<RuleType, RulesConditionMap*> _rule_condition_map;
};

}  // namespace idrc