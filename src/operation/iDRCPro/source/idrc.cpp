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

namespace idrc {

DrcManager::DrcManager()
{
  _data_manager = new DrcDataManager();
  // _rule_manager = new DrcRuleManager(_engine);
  _violation_manager = new DrcViolationManager();
  _condition_manager = new DrcConditionManager(_violation_manager);
  _engine = new DrcEngine(_data_manager, _condition_manager);
}

DrcManager::~DrcManager()
{
  // if (_rule_manager != nullptr) {
  //   delete _rule_manager;
  //   _rule_manager = nullptr;
  // }

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
  _engine->get_engine_manager()->filterData();  // TODO: self violation and intersection violation
  return true;
}

void DrcManager::check()
{
  _engine->get_engine_manager()->get_engine_check()->check();
}

}  // namespace idrc