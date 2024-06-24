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

#include "RTHeader.hpp"
#include "TGNet.hpp"
#include "TGNode.hpp"
#include "TGParameter.hpp"

namespace irt {

class TGModel
{
 public:
  TGModel() = default;
  ~TGModel() = default;
  // getter
  std::vector<TGNet>& get_tg_net_list() { return _tg_net_list; }
  TGParameter& get_tg_parameter() { return _tg_parameter; }
  std::vector<TGNet*>& get_tg_task_list() { return _tg_task_list; }
  GridMap<TGNode>& get_tg_node_map() { return _tg_node_map; }
  // setter
  void set_tg_net_list(const std::vector<TGNet>& tg_net_list) { _tg_net_list = tg_net_list; }
  void set_tg_parameter(const TGParameter& tg_parameter) { _tg_parameter = tg_parameter; }
  void set_tg_task_list(const std::vector<TGNet*>& tg_task_list) { _tg_task_list = tg_task_list; }
  void set_tg_node_map(const GridMap<TGNode>& tg_node_map) { _tg_node_map = tg_node_map; }

 private:
  std::vector<TGNet> _tg_net_list;
  TGParameter _tg_parameter;
  std::vector<TGNet*> _tg_task_list;
  GridMap<TGNode> _tg_node_map;
};

}  // namespace irt
