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
#include "DRNet.hpp"
#include "GridMap.hpp"
#include "DRParameter.hpp"

namespace irt {

class DRModel
{
 public:
  DRModel() = default;
  ~DRModel() = default;
  // getter
  std::vector<DRNet>& get_dr_net_list() { return _dr_net_list; }
  DRParameter& get_curr_dr_parameter() { return _curr_dr_parameter; }
  GridMap<DRBox>& get_dr_box_map() { return _dr_box_map; }
  std::vector<std::vector<DRBoxId>>& get_dr_box_id_list_list() { return _dr_box_id_list_list; }
  // setter
  void set_dr_net_list(const std::vector<DRNet>& dr_net_list) { _dr_net_list = dr_net_list; }
  void set_curr_dr_parameter(const DRParameter& curr_dr_parameter) { _curr_dr_parameter = curr_dr_parameter; }
  void set_dr_box_map(const GridMap<DRBox>& dr_box_map) { _dr_box_map = dr_box_map; }
  void set_dr_box_id_list_list(const std::vector<std::vector<DRBoxId>>& dr_box_id_list_list) { _dr_box_id_list_list = dr_box_id_list_list; }

 private:
  std::vector<DRNet> _dr_net_list;
  // iter
  DRParameter _curr_dr_parameter;
  GridMap<DRBox> _dr_box_map;
  std::vector<std::vector<DRBoxId>> _dr_box_id_list_list;
};

}  // namespace irt
