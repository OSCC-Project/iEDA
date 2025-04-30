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

#include "DRBoxId.hpp"
#include "DRIterParam.hpp"
#include "DRNet.hpp"
#include "GridMap.hpp"

namespace irt {

class DRModel
{
 public:
  DRModel() = default;
  ~DRModel() = default;
  // getter
  std::vector<DRNet>& get_dr_net_list() { return _dr_net_list; }
  bool get_initial_routing() const { return _initial_routing; }
  int32_t get_iter() const { return _iter; }
  DRIterParam& get_dr_iter_param() { return _dr_iter_param; }
  GridMap<DRBox>& get_dr_box_map() { return _dr_box_map; }
  std::vector<std::vector<DRBoxId>>& get_dr_box_id_list_list() { return _dr_box_id_list_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_detailed_result_map() { return _best_net_detailed_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_best_net_detailed_patch_map() { return _best_net_detailed_patch_map; }
  std::vector<Violation>& get_best_route_violation_list() { return _best_route_violation_list; }
  // setter
  void set_dr_net_list(const std::vector<DRNet>& dr_net_list) { _dr_net_list = dr_net_list; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_dr_iter_param(const DRIterParam& dr_iter_param) { _dr_iter_param = dr_iter_param; }
  void set_dr_box_map(const GridMap<DRBox>& dr_box_map) { _dr_box_map = dr_box_map; }
  void set_dr_box_id_list_list(const std::vector<std::vector<DRBoxId>>& dr_box_id_list_list) { _dr_box_id_list_list = dr_box_id_list_list; }
  void set_best_net_detailed_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_detailed_result_map)
  {
    _best_net_detailed_result_map = best_net_detailed_result_map;
  }
  void set_best_net_detailed_patch_map(const std::map<int32_t, std::vector<EXTLayerRect>>& best_net_detailed_patch_map)
  {
    _best_net_detailed_patch_map = best_net_detailed_patch_map;
  }
  void set_best_route_violation_list(const std::vector<Violation>& best_route_violation_list) { _best_route_violation_list = best_route_violation_list; }

 private:
  std::vector<DRNet> _dr_net_list;
  bool _initial_routing = true;
  int32_t _iter = -1;
  DRIterParam _dr_iter_param;
  GridMap<DRBox> _dr_box_map;
  std::vector<std::vector<DRBoxId>> _dr_box_id_list_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_detailed_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _best_net_detailed_patch_map;
  std::vector<Violation> _best_route_violation_list;
};

}  // namespace irt
