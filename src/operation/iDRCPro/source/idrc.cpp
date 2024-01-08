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
#include "idrc.h"

#include "condition_builder.h"
#include "condition_connectivity.h"
#include "condition_eol.h"
#include "condition_spacing_table.h"
#include "condition_step.h"

namespace idrc {

DrcManager::DrcManager()
{
  _data_manager = new DrcDataManager();
  _engine = new DrcEngine(_data_manager);
  _rule_manager = new DrcRuleManager(_engine);
  _violation_manager = new DrcViolationManager();
  _condition_manager = new DrcConditionManager(_engine, _violation_manager);
}

DrcManager::~DrcManager()
{
  if (_rule_manager != nullptr) {
    delete _rule_manager;
    _rule_manager = nullptr;
  }

  if (_condition_manager != nullptr) {
    delete _condition_manager;
    _condition_manager = nullptr;
  }

  if (_engine != nullptr) {
    delete _engine;
    _engine = nullptr;
  }

  if (_data_manager != nullptr) {
    delete _data_manager;
    _data_manager = nullptr;
  }

  if (_violation_manager != nullptr) {
    delete _violation_manager;
    _violation_manager = nullptr;
  }
}

void DrcManager::init(std::string config)
{
}

void DrcManager::engineStart(DrcCheckerType checker_type)
{
  _engine->initEngine(checker_type);
}
/**
 * return true : has conditon to check
 * return false : not condition need to check
 */
bool DrcManager::buildCondition()
{
  DrcConditionBuilder builder(_condition_manager);

  return builder.buildCondition();
}

void DrcManager::check()
{
  // TODO: sratagy and multi-thread

  DrcRuleConditionSpacingTable spacing_table(_condition_manager, _engine);

  spacing_table.checkFastMode();

  DrcRuleConditionConnectivity connectivity(_condition_manager, _engine);

  connectivity.checkFastMode();

  DrcRuleConditionStep condition_step(_condition_manager, _engine);

  condition_step.checkFastMode();

  DrcRuleConditionEOL condition_eol(_condition_manager, _engine);

  condition_eol.checkFastMode();
}

}  // namespace idrc