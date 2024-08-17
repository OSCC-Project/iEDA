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

#include "PANet.hpp"
#include "RTHeader.hpp"
#include "PAParameter.hpp"
#include "PABox.hpp"

namespace irt {

class PAModel
{
 public:
  PAModel() = default;
  ~PAModel() = default;
  // getter
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  PAParameter& get_pa_parameter() { return _pa_parameter; }
  GridMap<PABox>& get_pa_box_map() { return _pa_box_map; }
  std::vector<std::vector<PABoxId>>& get_pa_box_id_list_list() { return _pa_box_id_list_list; }
  // setter
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_pa_parameter(const PAParameter& pa_parameter) { _pa_parameter = pa_parameter; }
  void set_pa_box_map(const GridMap<PABox>& pa_box_map) { _pa_box_map = pa_box_map; }
  void set_pa_box_id_list_list(const std::vector<std::vector<PABoxId>>& pa_box_id_list_list) { _pa_box_id_list_list = pa_box_id_list_list; }

 private:
  std::vector<PANet> _pa_net_list;
  PAParameter _pa_parameter;
  GridMap<PABox> _pa_box_map;
  std::vector<std::vector<PABoxId>> _pa_box_id_list_list;
};

}  // namespace irt
