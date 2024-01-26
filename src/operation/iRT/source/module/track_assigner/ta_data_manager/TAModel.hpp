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

#include "TANet.hpp"
#include "TAPanel.hpp"
#include "TAPanelId.hpp"

namespace irt {

class TAModel
{
 public:
  TAModel() = default;
  ~TAModel() = default;
  // getter
  std::vector<TANet>& get_ta_net_list() { return _ta_net_list; }
  TAParameter& get_curr_ta_parameter() { return _curr_ta_parameter; }
  std::vector<std::vector<TAPanel>>& get_layer_panel_list() { return _layer_panel_list; }
  // setter
  void set_ta_net_list(const std::vector<TANet>& ta_net_list) { _ta_net_list = ta_net_list; }
  void set_curr_ta_parameter(const TAParameter& curr_ta_parameter) { _curr_ta_parameter = curr_ta_parameter; }
  void set_layer_panel_list(const std::vector<std::vector<TAPanel>>& layer_panel_list) { _layer_panel_list = layer_panel_list; }

 private:
  std::vector<TANet> _ta_net_list;
  TAParameter _curr_ta_parameter;
  std::vector<std::vector<TAPanel>> _layer_panel_list;
};

}  // namespace irt
