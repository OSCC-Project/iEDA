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
#include "VRGCell.hpp"
#include "VRModelStat.hpp"
#include "VRNet.hpp"

namespace irt {

class VRModel
{
 public:
  VRModel() = default;
  ~VRModel() = default;
  // getter
  GridMap<VRGCell>& get_vr_gcell_map() { return _vr_gcell_map; }
  std::vector<VRNet>& get_vr_net_list() { return _vr_net_list; }
  VRModelStat& get_vr_model_stat() { return _vr_model_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_vr_gcell_map(const GridMap<VRGCell>& vr_gcell_map) { _vr_gcell_map = vr_gcell_map; }
  void set_vr_net_list(const std::vector<VRNet>& vr_net_list) { _vr_net_list = vr_net_list; }
  void set_vr_model_stat(const VRModelStat& vr_model_stat) { _vr_model_stat = vr_model_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }

 private:
  GridMap<VRGCell> _vr_gcell_map;
  std::vector<VRNet> _vr_net_list;
  VRModelStat _vr_model_stat;
  irt_int _curr_iter = -1;
};

}  // namespace irt
