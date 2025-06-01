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

#include "GridMap.hpp"
#include "SRBoxId.hpp"
#include "SRIterParam.hpp"
#include "SRNet.hpp"

namespace irt {

class SRModel
{
 public:
  SRModel() = default;
  ~SRModel() = default;
  // getter
  std::vector<SRNet>& get_sr_net_list() { return _sr_net_list; }
  bool get_initial_routing() const { return _initial_routing; }
  std::vector<GridMap<SRNode>>& get_layer_node_map() { return _layer_node_map; }
  int32_t get_iter() const { return _iter; }
  SRIterParam& get_sr_iter_param() { return _sr_iter_param; }
  GridMap<SRBox>& get_sr_box_map() { return _sr_box_map; }
  std::vector<std::vector<SRBoxId>>& get_sr_box_id_list_list() { return _sr_box_id_list_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_task_global_result_map() { return _best_net_task_global_result_map; }
  double get_best_overflow() const { return _best_overflow; }
  // setter
  void set_sr_net_list(const std::vector<SRNet>& sr_net_list) { _sr_net_list = sr_net_list; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_layer_node_map(const std::vector<GridMap<SRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_sr_iter_param(const SRIterParam& sr_iter_param) { _sr_iter_param = sr_iter_param; }
  void set_sr_box_map(const GridMap<SRBox>& sr_box_map) { _sr_box_map = sr_box_map; }
  void set_sr_box_id_list_list(const std::vector<std::vector<SRBoxId>>& sr_box_id_list_list) { _sr_box_id_list_list = sr_box_id_list_list; }
  void set_best_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map)
  {
    _best_net_task_global_result_map = best_net_task_global_result_map;
  }
  void set_best_overflow(const double best_overflow) { _best_overflow = best_overflow; }

 private:
  std::vector<SRNet> _sr_net_list;
  bool _initial_routing = true;
  std::vector<GridMap<SRNode>> _layer_node_map;
  int32_t _iter = -1;
  SRIterParam _sr_iter_param;
  GridMap<SRBox> _sr_box_map;
  std::vector<std::vector<SRBoxId>> _sr_box_id_list_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_task_global_result_map;
  double _best_overflow = 0;
};

}  // namespace irt
