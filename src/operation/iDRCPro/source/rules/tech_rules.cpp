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

#include "tech_rules.h"

#include "rule_builder.h"

namespace idrc {

TechRules* TechRules::_instance = nullptr;

void TechRules::destroyInst()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

TechRules::~TechRules()
{
  for (auto& [layer, condition_map] : _condition_routing_layers) {
    for (auto& condition_list : condition_map) {
      for (auto& condition : condition_list.second) {
        delete condition;
      }
    }
  }
}

void TechRules::init()
{
  DrcRuleBuilder builder;
  builder.build();
}

}  // namespace idrc