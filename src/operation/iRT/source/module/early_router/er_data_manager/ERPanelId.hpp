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

#include "RTHeader.hpp"

namespace irt {

class ERPanelId
{
 public:
  ERPanelId() = default;
  ERPanelId(const int32_t layer_idx, const int32_t panel_idx)
  {
    _layer_idx = layer_idx;
    _panel_idx = panel_idx;
  }
  ~ERPanelId() = default;
  bool operator==(const ERPanelId& other) { return this->_layer_idx == other._layer_idx && this->_panel_idx == other._panel_idx; }
  bool operator!=(const ERPanelId& other) { return !((*this) == other); }
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_panel_idx() const { return _panel_idx; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_panel_idx(const int32_t panel_idx) { _panel_idx = panel_idx; }
  // function

 private:
  int32_t _layer_idx = -1;
  int32_t _panel_idx = -1;
};

struct CmpERPanelId
{
  bool operator()(const ERPanelId& a, const ERPanelId& b) const
  {
    if (a.get_layer_idx() != b.get_layer_idx()) {
      return a.get_layer_idx() < b.get_layer_idx();
    } else {
      return a.get_panel_idx() < b.get_panel_idx();
    }
  }
};

}  // namespace irt
