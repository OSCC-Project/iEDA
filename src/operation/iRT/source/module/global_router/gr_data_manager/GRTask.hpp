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

#include "GRGroup.hpp"

namespace irt {

class GRTask
{
 public:
  GRTask() = default;
  ~GRTask() = default;
  // getter
  std::vector<GRGroup>& get_gr_group_list() { return _gr_group_list; }
  // setter
  void set_gr_group_list(const std::vector<GRGroup>& gr_group_list) { _gr_group_list = gr_group_list; }
  // function

 private:
  std::vector<GRGroup> _gr_group_list;
};

}  // namespace irt
