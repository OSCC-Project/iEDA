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

#include "LAComParam.hpp"
#include "LANet.hpp"
#include "LANode.hpp"

namespace irt {

class LAModel
{
 public:
  LAModel() = default;
  ~LAModel() = default;
  // getter
  std::vector<LANet>& get_la_net_list() { return _la_net_list; }
  LAComParam& get_la_com_param() { return _la_com_param; }
  std::vector<LANet*>& get_la_task_list() { return _la_task_list; }
  std::vector<GridMap<LANode>>& get_layer_node_map() { return _layer_node_map; }
  // setter
  void set_la_net_list(const std::vector<LANet>& la_net_list) { _la_net_list = la_net_list; }
  void set_la_com_param(const LAComParam& la_com_param) { _la_com_param = la_com_param; }
  void set_la_task_list(const std::vector<LANet*>& la_task_list) { _la_task_list = la_task_list; }
  void set_layer_node_map(const std::vector<GridMap<LANode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  // function

#if 1
  // single task
  LANet* get_curr_la_task() { return _curr_la_task; }
  void set_curr_la_task(LANet* curr_la_task) { _curr_la_task = curr_la_task; }
#endif

 private:
  std::vector<LANet> _la_net_list;
  LAComParam _la_com_param;
  std::vector<LANet*> _la_task_list;
  std::vector<GridMap<LANode>> _layer_node_map;
#if 1
  // single task
  LANet* _curr_la_task = nullptr;
#endif
};

}  // namespace irt
