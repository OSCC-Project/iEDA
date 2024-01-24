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
#include <queue>
#include <vector>

#include "check_item.h"
#include "condition.h"

namespace idrc {

class DrcEngineCheck
{
 public:
  DrcEngineCheck() {}
  ~DrcEngineCheck() {}

  void addCheckItem(Condition* condition, CheckItem* check_item) { _check_list[condition].push_back(check_item); }

  void check() { applyCondition(); }

 private:
  std::map<Condition*, std::deque<CheckItem*>> _check_list;

  void applyCondition()
  {
    // for (auto& [condition, queue] : _check_list) {  // TODO: while(1), thread pool
    //   while (!queue.empty()) {
    //     auto* item = queue.front();
    //     queue.pop_front();
    //     condition->get_detail()->apply(item);
    //     delete item;
    //   }
    // }
  }
};

}  // namespace idrc