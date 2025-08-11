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

#include "PABox.hpp"
#include "PAComParam.hpp"
#include "PAIterParam.hpp"
#include "PANet.hpp"
#include "RTHeader.hpp"

namespace irt {

class PAModel
{
 public:
  PAModel() = default;
  ~PAModel() = default;
  // getter
  PAComParam& get_pa_com_param() { return _pa_com_param; }
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  bool get_initial_routing() const { return _initial_routing; }
  int32_t get_iter() const { return _iter; }
  PAIterParam& get_pa_iter_param() { return _pa_iter_param; }
  GridMap<PABox>& get_pa_box_map() { return _pa_box_map; }
  std::vector<std::vector<PABoxId>>& get_pa_box_id_list_list() { return _pa_box_id_list_list; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_best_net_pin_access_result_map() { return _best_net_pin_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& get_best_net_pin_access_patch_map() { return _best_net_pin_access_patch_map; }
  std::vector<Violation>& get_best_route_violation_list() { return _best_route_violation_list; }
  // setter
  void set_pa_com_param(const PAComParam& pa_com_param) { _pa_com_param = pa_com_param; }
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_pa_iter_param(const PAIterParam& pa_iter_param) { _pa_iter_param = pa_iter_param; }
  void set_pa_box_map(const GridMap<PABox>& pa_box_map) { _pa_box_map = pa_box_map; }
  void set_pa_box_id_list_list(const std::vector<std::vector<PABoxId>>& pa_box_id_list_list) { _pa_box_id_list_list = pa_box_id_list_list; }
  void set_best_net_pin_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& best_net_pin_access_result_map)
  {
    _best_net_pin_access_result_map = best_net_pin_access_result_map;
  }
  void set_best_net_pin_access_patch_map(const std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& best_net_pin_access_patch_map)
  {
    _best_net_pin_access_patch_map = best_net_pin_access_patch_map;
  }
  void set_best_route_violation_list(const std::vector<Violation>& best_route_violation_list) { _best_route_violation_list = best_route_violation_list; }

 private:
  PAComParam _pa_com_param;
  std::vector<PANet> _pa_net_list;
  bool _initial_routing = true;
  int32_t _iter = -1;
  PAIterParam _pa_iter_param;
  GridMap<PABox> _pa_box_map;
  std::vector<std::vector<PABoxId>> _pa_box_id_list_list;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _best_net_pin_access_result_map;
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>> _best_net_pin_access_patch_map;
  std::vector<Violation> _best_route_violation_list;
};

}  // namespace irt
