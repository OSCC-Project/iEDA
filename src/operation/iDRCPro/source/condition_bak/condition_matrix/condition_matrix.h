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
#include "condition_manager.h"
#include "idrc_engine.h"
#include "idrc_rule_stratagy.h"
#include "rule_enum.h"

namespace idrc {

/**
 * rule conditions are concepts built from tech lef drc rules, it contains a condition matrix to guide condition check orders, the rule
 * matrix index indicates the checking order,
 *
 */

class DrcRuleCondition;

class DrcRuleConditionMatrix
{
 public:
  DrcRuleConditionMatrix(DrcConditionManager* condition_manager, DrcEngine* engine, RuleType type)
      : _condition_manager(condition_manager), _engine(engine), _type(type)
  {
  }
  ~DrcRuleConditionMatrix() {}

  DrcEngine* get_engine() { return _engine; }
  RuleType get_type() { return _type; }

  bool check(DrcStratagyType type = DrcStratagyType::kCheckComplete);
  virtual bool checkFastMode() = 0;
  virtual bool checkCompleteMode() = 0;

 protected:
  DrcConditionManager* _condition_manager = nullptr;
  DrcEngine* _engine = nullptr;

 private:
  RuleType _type;  // check conditon type
  //   DrcRuleCondition* _condition = nullptr;  /// transfer condition data from stage 1 to stage n...
};

}  // namespace idrc