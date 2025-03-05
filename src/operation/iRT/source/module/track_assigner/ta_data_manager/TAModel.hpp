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
  TAComParam& get_ta_com_param() { return _ta_com_param; }
  std::vector<std::vector<TAPanel>>& get_layer_panel_list() { return _layer_panel_list; }
  std::vector<std::vector<TAPanelId>>& get_ta_panel_id_list_list() { return _ta_panel_id_list_list; }
  // setter
  void set_ta_net_list(const std::vector<TANet>& ta_net_list) { _ta_net_list = ta_net_list; }
  void set_ta_com_param(const TAComParam& ta_com_param) { _ta_com_param = ta_com_param; }
  void set_layer_panel_list(const std::vector<std::vector<TAPanel>>& layer_panel_list) { _layer_panel_list = layer_panel_list; }
  void set_ta_panel_id_list_list(const std::vector<std::vector<TAPanelId>>& ta_panel_id_list_list) { _ta_panel_id_list_list = ta_panel_id_list_list; }

 private:
  std::vector<TANet> _ta_net_list;
  TAComParam _ta_com_param;
  std::vector<std::vector<TAPanel>> _layer_panel_list;
  std::vector<std::vector<TAPanelId>> _ta_panel_id_list_list;
};

}  // namespace irt
