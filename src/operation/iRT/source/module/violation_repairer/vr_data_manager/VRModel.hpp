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
#include "VRBox.hpp"
#include "VRIterParam.hpp"
#include "VRNet.hpp"

namespace irt {

class VRModel
{
 public:
  VRModel() = default;
  ~VRModel() = default;
  // getter
  std::vector<VRNet>& get_vr_net_list() { return _vr_net_list; }
  int32_t get_iter() const { return _iter; }
  VRIterParam& get_vr_iter_param() { return _vr_iter_param; }
  GridMap<VRBox>& get_vr_box_map() { return _vr_box_map; }
  std::vector<std::vector<VRBoxId>>& get_vr_box_id_list_list() { return _vr_box_id_list_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_final_result_map() { return _best_net_final_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_best_net_final_patch_map() { return _best_net_final_patch_map; }
  std::vector<Violation>& get_best_violation_list() { return _best_violation_list; }
  // setter
  void set_vr_net_list(const std::vector<VRNet>& vr_net_list) { _vr_net_list = vr_net_list; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_vr_iter_param(const VRIterParam& vr_iter_param) { _vr_iter_param = vr_iter_param; }
  void set_vr_box_map(const GridMap<VRBox>& vr_box_map) { _vr_box_map = vr_box_map; }
  void set_vr_box_id_list_list(const std::vector<std::vector<VRBoxId>>& vr_box_id_list_list) { _vr_box_id_list_list = vr_box_id_list_list; }
  void set_best_net_final_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_final_result_map)
  {
    _best_net_final_result_map = best_net_final_result_map;
  }
  void set_best_net_final_patch_map(const std::map<int32_t, std::vector<EXTLayerRect>>& best_net_final_patch_map)
  {
    _best_net_final_patch_map = best_net_final_patch_map;
  }
  void set_best_violation_list(const std::vector<Violation>& best_violation_list) { _best_violation_list = best_violation_list; }

 private:
  std::vector<VRNet> _vr_net_list;
  int32_t _iter = -1;
  VRIterParam _vr_iter_param;
  GridMap<VRBox> _vr_box_map;
  std::vector<std::vector<VRBoxId>> _vr_box_id_list_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_final_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _best_net_final_patch_map;
  std::vector<Violation> _best_violation_list;
};

}  // namespace irt
