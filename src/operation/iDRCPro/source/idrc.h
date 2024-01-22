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

#include <string>

#include "condition_manager.h"
#include "idrc_config.h"
#include "idrc_data.h"
#include "idrc_dm.h"
#include "idrc_engine.h"
#include "idrc_violation_manager.h"

namespace idrc {

class DrcManager
{
 public:
  DrcManager();
  ~DrcManager();

  DrcDataManager* get_data_manager() { return _data_manager; }
  // DrcRuleManager* get_rule_manager() { return _rule_manager; }
  DrcConditionManager* get_condition_manager() { return _condition_manager; }
  DrcViolationManager* get_violation_manager() { return _violation_manager; }
  DrcEngine* get_engine() { return _engine; }

  void init(std::string config = "");
  void engineStart(DrcCheckerType checker_type = DrcCheckerType::kRT);
  bool buildCondition();
  void check();
  void checkSelf();

 private:
  DrcDataManager* _data_manager;
  // DrcRuleManager* _rule_manager;
  DrcConditionManager* _condition_manager = nullptr;
  DrcViolationManager* _violation_manager;
  DrcEngine* _engine;
};

}  // namespace idrc