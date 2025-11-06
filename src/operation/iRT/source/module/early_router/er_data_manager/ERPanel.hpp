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

#include "ERPanelId.hpp"
#include "LayerRect.hpp"
#include "OpenQueue.hpp"
#include "RTHeader.hpp"
#include "ScaleAxis.hpp"
#include "TAComParam.hpp"
#include "TANode.hpp"

namespace irt {

class ERPanel
{
 public:
  ERPanel() = default;
  ~ERPanel() = default;
  // getter
  EXTLayerRect& get_panel_rect() { return _panel_rect; }
  ERPanelId& get_er_panel_id() { return _er_panel_id; }
  // setter
  void set_panel_rect(const EXTLayerRect& panel_rect) { _panel_rect = panel_rect; }
  void set_er_panel_id(const ERPanelId& er_panel_id) { _er_panel_id = er_panel_id; }
  // function

 private:
  EXTLayerRect _panel_rect;
  ERPanelId _er_panel_id;
};

}  // namespace irt