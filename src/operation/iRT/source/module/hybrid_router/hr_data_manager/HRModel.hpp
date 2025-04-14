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
#include "HRBoxId.hpp"
#include "HRIterParam.hpp"
#include "HRNet.hpp"

namespace irt {

class HRModel
{
 public:
  HRModel() = default;
  ~HRModel() = default;
  // getter
  std::vector<HRNet>& get_hr_net_list() { return _hr_net_list; }
  bool get_initial_routing() const { return _initial_routing; }
  int32_t get_iter() const { return _iter; }
  HRIterParam& get_hr_iter_param() { return _hr_iter_param; }
  GridMap<HRBox>& get_hr_box_map() { return _hr_box_map; }
  std::vector<std::vector<HRBoxId>>& get_hr_box_id_list_list() { return _hr_box_id_list_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_final_result_map() { return _best_net_final_result_map; }
  std::vector<Violation>& get_best_violation_list() { return _best_violation_list; }
  // setter
  void set_hr_net_list(const std::vector<HRNet>& hr_net_list) { _hr_net_list = hr_net_list; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_hr_iter_param(const HRIterParam& hr_iter_param) { _hr_iter_param = hr_iter_param; }
  void set_hr_box_map(const GridMap<HRBox>& hr_box_map) { _hr_box_map = hr_box_map; }
  void set_hr_box_id_list_list(const std::vector<std::vector<HRBoxId>>& hr_box_id_list_list) { _hr_box_id_list_list = hr_box_id_list_list; }
  void set_best_net_final_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_final_result_map)
  {
    _best_net_final_result_map = best_net_final_result_map;
  }
  void set_best_violation_list(const std::vector<Violation>& best_violation_list) { _best_violation_list = best_violation_list; }

 private:
  std::vector<HRNet> _hr_net_list;
  bool _initial_routing = true;
  int32_t _iter = -1;
  HRIterParam _hr_iter_param;
  GridMap<HRBox> _hr_box_map;
  std::vector<std::vector<HRBoxId>> _hr_box_id_list_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_final_result_map;
  std::vector<Violation> _best_violation_list;
};

}  // namespace irt
