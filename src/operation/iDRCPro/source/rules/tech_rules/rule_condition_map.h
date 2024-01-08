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

#include <limits>
#include <map>
#include <vector>

#include "rule_basic.h"

namespace idrc {

class RulesConditionMap
{
 public:
  RulesConditionMap(RuleType type) : _type(type) {}
  virtual ~RulesConditionMap() {}

  std::map<RuleType, std::map<int, std::vector<ConditionRule*>, std::less<int>>>& get_conditon_rules() { return _conditon_rules; }
  std::map<int, std::vector<ConditionRule*>, std::less<int>>& get_rule_map(RuleType type) { return _conditon_rules[type]; }
  std::vector<ConditionRule*> get_condition_rule(RuleType type, int value) { return _conditon_rules[type][value]; }
  int get_min() { return _min; }
  int get_max() { return _max; }
  int get_default() { return _default_rule.first; }
  std::pair<int, ConditionRule*>& get_default_rule() { return _default_rule; }

  void set_condition_rule(RuleType type, int value, ConditionRule* rule)
  {
    _conditon_rules[type][value].emplace_back(rule);
    _min = std::min(_min, value);
    _max = std::max(_max, value);
  }
  void set_default_rule(int value, ConditionRule* rule)
  {
    _default_rule.first = value;
    _default_rule.second = rule;
  }

  /**
   * the value of map list are incremental
   * get the adjacent greater one rule which value
   *     rule1 value1 < value < rule1 value 2 --> select rule1 value2's rule
   *     rule2 value1 < value < rule2 vaule 2 --> select rule2 value2's rule
   *     ....
   * all rules must at least compare with the value one time
   */
  std::map<int, std::vector<ConditionRule*>> selectRuleGreater(RuleType type, int value)
  {
    std::map<int, std::vector<ConditionRule*>> rule_list;

    auto& rule_map = get_rule_map(type);
    for (auto& rule : rule_map) {
      /// value must be less than rule's value
      if (value <= rule.first) {
        rule_list[rule.first] = rule.second;

        /// check next rule in the map
        break;
      }
    }

    return rule_list;
  }

  /**
   * the value of map list are incremental
   * get the adjacent less one rule which value
   *     rule1 value1 < value < rule1 value 2 --> select rule1 value1's rule
   *     rule2 value1 < value < rule2 vaule 2 --> select rule2 value1's rule
   *     ....
   * all rules must at least compare with the value one time
   */
  std::map<int, std::vector<ConditionRule*>> selectRuleLess(RuleType type, int value)
  {
    std::map<int, std::vector<ConditionRule*>> rule_list;
    auto& rule_map = get_rule_map(type);
    /// reverse iterate
    for (std::map<int, std::vector<ConditionRule*>, std::less<int>>::reverse_iterator rule = rule_map.rbegin(); rule != rule_map.rend();
         ++rule) {
      /// value must be greater than rule's value
      if (value >= rule->first) {
        rule_list[rule->first] = rule->second;

        /// check next rule in the map
        break;
      }
    }
    return rule_list;
  }

 private:
  RuleType _type;
  std::map<RuleType, std::map<int, std::vector<ConditionRule*>, std::less<int>>>
      _conditon_rules;  /// int : value, ConditionRule* : condition rule for each layer

  std::pair<int, ConditionRule*> _default_rule{0, nullptr};  /// indicate the default rule

  int _min = std::numeric_limits<int>::max();
  int _max = std::numeric_limits<int>::min();
};

}  // namespace idrc