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

#include "ConflictAccessPoint.hpp"

namespace irt {

class ConflictGroup
{
 public:
  ConflictGroup() = default;
  ~ConflictGroup() = default;
  // getter
  std::vector<std::vector<ConflictAccessPoint>>& get_conflict_ap_list_list() { return _conflict_ap_list_list; }
  // setter
  void set_conflict_ap_list_list(const std::vector<std::vector<ConflictAccessPoint>>& conflict_ap_list_list)
  {
    _conflict_ap_list_list = conflict_ap_list_list;
  }
  // function
 private:
  std::vector<std::vector<ConflictAccessPoint>> _conflict_ap_list_list;
};

}  // namespace irt
