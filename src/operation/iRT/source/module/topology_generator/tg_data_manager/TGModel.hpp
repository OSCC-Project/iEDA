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
#include "TGComParam.hpp"
#include "TGNet.hpp"
#include "TGNode.hpp"

namespace irt {

class TGModel
{
 public:
  TGModel() = default;
  ~TGModel() = default;
  // getter
  std::vector<TGNet>& get_tg_net_list() { return _tg_net_list; }
  TGComParam& get_tg_com_param() { return _tg_com_param; }
  std::vector<TGNet*>& get_tg_task_list() { return _tg_task_list; }
  GridMap<TGNode>& get_tg_node_map() { return _tg_node_map; }
  // setter
  void set_tg_net_list(const std::vector<TGNet>& tg_net_list) { _tg_net_list = tg_net_list; }
  void set_tg_com_param(const TGComParam& tg_com_param) { _tg_com_param = tg_com_param; }
  void set_tg_task_list(const std::vector<TGNet*>& tg_task_list) { _tg_task_list = tg_task_list; }
  void set_tg_node_map(const GridMap<TGNode>& tg_node_map) { _tg_node_map = tg_node_map; }
#if 1
  // single task
  TGNet* get_curr_tg_task() { return _curr_tg_task; }
  void set_curr_tg_task(TGNet* curr_tg_task) { _curr_tg_task = curr_tg_task; }
#endif

 private:
  std::vector<TGNet> _tg_net_list;
  TGComParam _tg_com_param;
  std::vector<TGNet*> _tg_task_list;
  GridMap<TGNode> _tg_node_map;
#if 1
  // single task
  TGNet* _curr_tg_task = nullptr;
#endif
};

}  // namespace irt
