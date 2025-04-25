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

#include "DPBox.hpp"
#include "DPIterParam.hpp"
#include "DPNet.hpp"
#include "DPSolution.hpp"
#include "GridMap.hpp"

namespace irt {

class DPModel
{
 public:
  DPModel() = default;
  ~DPModel() = default;
  // getter
  std::vector<DPNet>& get_dp_net_list() { return _dp_net_list; }
  int32_t get_iter() const { return _iter; }
  DPIterParam& get_dp_iter_param() { return _dp_iter_param; }
  GridMap<DPBox>& get_dp_box_map() { return _dp_box_map; }
  std::vector<std::vector<DPBoxId>>& get_dp_box_id_list_list() { return _dp_box_id_list_list; }
  std::map<DPBoxId, std::map<int32_t, std::vector<EXTLayerRect>>, CmpDPBoxId>& get_box_net_force_patch_map() { return _box_net_force_patch_map; }
  // setter
  void set_dp_net_list(const std::vector<DPNet>& dp_net_list) { _dp_net_list = dp_net_list; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_dp_iter_param(const DPIterParam& dp_iter_param) { _dp_iter_param = dp_iter_param; }
  void set_dp_box_map(const GridMap<DPBox>& dp_box_map) { _dp_box_map = dp_box_map; }
  void set_dp_box_id_list_list(const std::vector<std::vector<DPBoxId>>& dp_box_id_list_list) { _dp_box_id_list_list = dp_box_id_list_list; }
  void set_box_net_force_patch_map(const std::map<DPBoxId, std::map<int32_t, std::vector<EXTLayerRect>>, CmpDPBoxId>& box_net_force_patch_map)
  {
    _box_net_force_patch_map = box_net_force_patch_map;
  }

 private:
  std::vector<DPNet> _dp_net_list;
  int32_t _iter = -1;
  DPIterParam _dp_iter_param;
  GridMap<DPBox> _dp_box_map;
  std::vector<std::vector<DPBoxId>> _dp_box_id_list_list;
  std::map<DPBoxId, std::map<int32_t, std::vector<EXTLayerRect>>, CmpDPBoxId> _box_net_force_patch_map;
};

}  // namespace irt
