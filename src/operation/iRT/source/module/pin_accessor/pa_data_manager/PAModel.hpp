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

#include "ConflictGroup.hpp"
#include "PANet.hpp"
#include "RTHeader.hpp"

namespace irt {

class PAModel
{
 public:
  PAModel() = default;
  ~PAModel() = default;
  // getter
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  std::vector<ConflictGroup>& get_conflict_group_list() { return _conflict_group_list; }
  // setter
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_conflict_group_list(const std::vector<ConflictGroup>& conflict_group_list) { _conflict_group_list = conflict_group_list; }

 private:
  std::vector<PANet> _pa_net_list;
  std::vector<ConflictGroup> _conflict_group_list;
};

}  // namespace irt
