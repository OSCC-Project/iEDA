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
#include "condition_matrix.h"
#include "idrc_engine.h"
#include "idrc_rule_stratagy.h"

namespace idrc {

/**
 * check notch conditions
 *
 */

class DrcRuleConditionNotch : public DrcRuleConditionMatrix
{
 public:
  DrcRuleConditionNotch(DrcConditionManager* condition_manager, DrcEngine* engine)
      : DrcRuleConditionMatrix(condition_manager, engine, RuleType::kEdgeNotch)
  {
  }
  ~DrcRuleConditionNotch() {}

  bool checkFastMode() override;
  bool checkCompleteMode() override;

 private:
  /// check
  bool checkNotch();  // Mode

  bool checkNotch(DrcBasicPoint* point_prev, DrcBasicPoint* point_next, idb::IdbLayer* layer,
                  std::map<int, idrc::ConditionRule*> rule_notch_map);
};

}  // namespace idrc