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
#include "DRModelStat.hpp"
#include "DRNet.hpp"
#include "GridMap.hpp"

namespace irt {

class DRModel
{
 public:
  DRModel() = default;
  ~DRModel() = default;
  // getter
  GridMap<DRBox>& get_dr_box_map() { return _dr_box_map; }
  std::vector<DRNet>& get_dr_net_list() { return _dr_net_list; }
  std::vector<std::vector<DRBoxId>>& get_dr_box_id_comb_list() { return _dr_box_id_comb_list; }
  DRModelStat& get_dr_model_stat() { return _dr_model_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_dr_box_map(const GridMap<DRBox>& dr_box_map) { _dr_box_map = dr_box_map; }
  void set_dr_net_list(const std::vector<DRNet>& dr_net_list) { _dr_net_list = dr_net_list; }
  void set_dr_box_id_comb_list(const std::vector<std::vector<DRBoxId>>& dr_box_id_comb_list) { _dr_box_id_comb_list = dr_box_id_comb_list; }
  void set_dr_model_stat(const DRModelStat& dr_model_stat) { _dr_model_stat = dr_model_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }

 private:
  GridMap<DRBox> _dr_box_map;
  std::vector<DRNet> _dr_net_list;
  std::vector<std::vector<DRBoxId>> _dr_box_id_comb_list;
  DRModelStat _dr_model_stat;
  irt_int _curr_iter = -1;
};

}  // namespace irt
