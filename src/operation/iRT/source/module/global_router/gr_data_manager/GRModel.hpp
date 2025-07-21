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

#include "GRBoxId.hpp"
#include "GRIterParam.hpp"
#include "GRNet.hpp"
#include "GridMap.hpp"

namespace irt {

class GRModel
{
 public:
  GRModel() = default;
  ~GRModel() = default;
  // getter
  std::vector<GRNet>& get_gr_net_list() { return _gr_net_list; }
  std::vector<GridMap<GRNode>>& get_layer_node_map() { return _layer_node_map; }
  int32_t get_iter() const { return _iter; }
  GRIterParam& get_gr_iter_param() { return _gr_iter_param; }
  GridMap<GRBox>& get_gr_box_map() { return _gr_box_map; }
  std::vector<std::vector<GRBoxId>>& get_gr_box_id_list_list() { return _gr_box_id_list_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_task_global_result_map() { return _best_net_task_global_result_map; }
  int32_t get_best_overflow() const { return _best_overflow; }
  // setter
  void set_gr_net_list(const std::vector<GRNet>& gr_net_list) { _gr_net_list = gr_net_list; }
  void set_layer_node_map(const std::vector<GridMap<GRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_gr_iter_param(const GRIterParam& gr_iter_param) { _gr_iter_param = gr_iter_param; }
  void set_gr_box_map(const GridMap<GRBox>& gr_box_map) { _gr_box_map = gr_box_map; }
  void set_gr_box_id_list_list(const std::vector<std::vector<GRBoxId>>& gr_box_id_list_list) { _gr_box_id_list_list = gr_box_id_list_list; }
  void set_best_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map)
  {
    _best_net_task_global_result_map = best_net_task_global_result_map;
  }
  void set_best_overflow(const int32_t best_overflow) { _best_overflow = best_overflow; }

 private:
  std::vector<GRNet> _gr_net_list;
  std::vector<GridMap<GRNode>> _layer_node_map;
  int32_t _iter = -1;
  GRIterParam _gr_iter_param;
  GridMap<GRBox> _gr_box_map;
  std::vector<std::vector<GRBoxId>> _gr_box_id_list_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_task_global_result_map;
  int32_t _best_overflow = 0;
};

}  // namespace irt
