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

#include "check_list.h"
#include "condition.h"

namespace idrc {

class DrcEngineCheck
{
 public:
  DrcEngineCheck() {}
  ~DrcEngineCheck()
  {
    for (auto& check_list : _check_list_map) {
      delete check_list.second;
    }
  }

  CheckList* get_check_list(Condition* condition)
  {
    Condition* base_condition = condition->get_base_condition() ? condition->get_base_condition() : condition;
    if (_check_list_map.find(base_condition) == _check_list_map.end()) {
      _check_list_map[base_condition] = new CheckList(base_condition);
    }
    return _check_list_map[base_condition];
  }

  void apply_condition_detail()  // TODO: parallel
  {
    for (auto& check_list : _check_list_map) {
      check_list.second->apply_condition_detail();
    }
  }

 private:
  std::map<Condition*, CheckList*> _check_list_map;
};

}  // namespace idrc