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

#include "PRNet.hpp"
#include "PRNode.hpp"
#include "PRParameter.hpp"
#include "RTHeader.hpp"

namespace irt {

class PRModel
{
 public:
  PRModel() = default;
  ~PRModel() = default;
  // getter
  std::vector<PRNet>& get_pr_net_list() { return _pr_net_list; }
  PRParameter& get_pr_parameter() { return _pr_parameter; }
  GridMap<PRNode>& get_pr_node_map() { return _pr_node_map; }
  std::vector<int32_t>& get_pr_net_idx_list() { return _pr_net_idx_list; }
  // setter
  void set_pr_net_list(const std::vector<PRNet>& pr_net_list) { _pr_net_list = pr_net_list; }
  void set_pr_parameter(const PRParameter& pr_parameter) { _pr_parameter = pr_parameter; }
  void set_pr_node_map(const GridMap<PRNode>& pr_node_map) { _pr_node_map = pr_node_map; }
  void set_pr_net_idx_list(const std::vector<int32_t>& pr_net_idx_list) { _pr_net_idx_list = pr_net_idx_list; }

 private:
  std::vector<PRNet> _pr_net_list;
  PRParameter _pr_parameter;
  GridMap<PRNode> _pr_node_map;
  std::vector<int32_t> _pr_net_idx_list;
};

}  // namespace irt
