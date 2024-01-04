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

class ConditionRuleMinStep : public ConditionRule
{
 public:
  ConditionRuleMinStep(RuleType type, int width, idb::IdbMinStep* min_step) : ConditionRule(type, width), _min_step(min_step) {}
  ~ConditionRuleMinStep() {}

  idb::IdbMinStep* get_min_step() { return _min_step; }

 private:
  idb::IdbMinStep* _min_step;
};

class ConditionRuleMinStepLef58 : public ConditionRule
{
 public:
  ConditionRuleMinStepLef58(RuleType type, int width, idb::routinglayer::Lef58MinStep* min_step)
      : ConditionRule(type, width), _min_step(min_step)
  {
  }
  ~ConditionRuleMinStepLef58() {}

  idb::routinglayer::Lef58MinStep* get_min_step() { return _min_step; }

 private:
  idb::routinglayer::Lef58MinStep* _min_step;
};

class ConditionRuleNotch : public ConditionRule
{
 public:
  ConditionRuleNotch(RuleType type, int spacing, idb::routinglayer::Lef58SpacingNotchlength* notch)
      : ConditionRule(type, spacing), _notch(notch)
  {
  }
  ~ConditionRuleNotch() {}

  idb::routinglayer::Lef58SpacingNotchlength* get_notch() { return _notch; }

 private:
  idb::routinglayer::Lef58SpacingNotchlength* _notch;
};

class ConditionRuleEOL : public ConditionRule
{
 public:
  ConditionRuleEOL(RuleType type, int spacing, idb::routinglayer::Lef58SpacingEol* eol) : ConditionRule(type, spacing), _eol(eol) {}
  ~ConditionRuleEOL() {}

  idb::routinglayer::Lef58SpacingEol* get_eol() { return _eol; }

 private:
  idb::routinglayer::Lef58SpacingEol* _eol;
};

class RulesMapEdge : public RulesConditionMap
{
 public:
  RulesMapEdge(RuleType type) : RulesConditionMap(type) {}
  ~RulesMapEdge() {}

 private:
};

}  // namespace idrc