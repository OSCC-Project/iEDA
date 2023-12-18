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

#include <map>

#include "BaseShape.hpp"

namespace irt {

class BaseInfo
{
 public:
  BaseInfo() = default;
  ~BaseInfo() = default;
  // getter
  int32_t get_net_idx() const { return _net_idx; }
  // pa
  int32_t get_pa_pin_idx() const { return _pa_pin_idx; }
  // ta
  int32_t get_ta_layer_idx() const { return _ta_layer_idx; }
  int32_t get_ta_panel_idx() const { return _ta_panel_idx; }
  int32_t get_ta_task_idx() const { return _ta_task_idx; }
  // dr
  int32_t get_dr_box_x() const { return _dr_box_x; }
  int32_t get_dr_box_y() const { return _dr_box_y; }
  int32_t get_dr_task_idx() const { return _dr_task_idx; }

  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  // pa
  void set_pa_pin_idx(const int32_t pa_pin_idx) { _pa_pin_idx = pa_pin_idx; }
  // ta
  void set_ta_layer_idx(const int32_t ta_layer_idx) { _ta_layer_idx = ta_layer_idx; }
  void set_ta_panel_idx(const int32_t ta_panel_idx) { _ta_panel_idx = ta_panel_idx; }
  void set_ta_task_idx(const int32_t ta_task_idx) { _ta_task_idx = ta_task_idx; }
  // dr
  void set_dr_box_x(const int32_t dr_box_x) { _dr_box_x = dr_box_x; }
  void set_dr_box_y(const int32_t dr_box_y) { _dr_box_y = dr_box_y; }
  void set_dr_task_idx(const int32_t dr_task_idx) { _dr_task_idx = dr_task_idx; }
  // function

 private:
  int32_t _net_idx = -1;
  // pa
  int32_t _pa_pin_idx = -1;
  // ta
  int32_t _ta_layer_idx = -1;
  int32_t _ta_panel_idx = -1;
  int32_t _ta_task_idx = -1;
  // dr
  int32_t _dr_box_x = -1;
  int32_t _dr_box_y = -1;
  int32_t _dr_task_idx = -1;
};

struct CmpBaseInfo
{
  bool operator()(const BaseInfo& a, const BaseInfo& b) const
  {
    if (a.get_net_idx() != b.get_net_idx()) {
      return a.get_net_idx() < b.get_net_idx();
    } else if (a.get_pa_pin_idx() != b.get_pa_pin_idx()) {
      return a.get_pa_pin_idx() < b.get_pa_pin_idx();
    } else if (a.get_ta_layer_idx() != b.get_ta_layer_idx()) {
      return a.get_ta_layer_idx() < b.get_ta_layer_idx();
    } else if (a.get_ta_panel_idx() != b.get_ta_panel_idx()) {
      return a.get_ta_panel_idx() < b.get_ta_panel_idx();
    } else if (a.get_ta_task_idx() != b.get_ta_task_idx()) {
      return a.get_ta_task_idx() < b.get_ta_task_idx();
    } else if (a.get_dr_box_x() != b.get_dr_box_x()) {
      return a.get_dr_box_x() < b.get_dr_box_x();
    } else if (a.get_dr_box_y() != b.get_dr_box_y()) {
      return a.get_dr_box_y() < b.get_dr_box_y();
    } else {
      return a.get_dr_task_idx() < b.get_dr_task_idx();
    }
  }
};

}  // namespace irt
