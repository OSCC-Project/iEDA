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
#include <set>
#include <string>
#include <vector>

namespace ids {

struct Shape
{
  int32_t net_idx = -1;
  int32_t ll_x = -1;
  int32_t ll_y = -1;
  int32_t ur_x = -1;
  int32_t ur_y = -1;
  int32_t layer_idx = -1;
  bool is_routing = true;
};

struct Violation
{
  std::string violation_type = "";
  int32_t ll_x = -1;
  int32_t ll_y = -1;
  int32_t ur_x = -1;
  int32_t ur_y = -1;
  int32_t layer_idx = -1;
  bool is_routing = true;
  std::set<int32_t> violation_net_set;
  int32_t required_size = 0;
};

}  // namespace ids
