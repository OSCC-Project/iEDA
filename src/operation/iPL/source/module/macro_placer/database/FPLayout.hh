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

#include "FPRect.hh"

namespace ipl::imp {

class FPLayout
{
 public:
  FPLayout();
  ~FPLayout();

  // getter
  FPRect* get_die_shape() const { return _die_shape; }
  FPRect* get_core_shape() const { return _core_shape; }

  // setter
  void set_die_x(int32_t x) { _die_shape->set_x(x); }
  void set_die_y(int32_t y) { _die_shape->set_y(y); }
  void set_die_width(uint32_t width) { _die_shape->set_width(width); }
  void set_die_height(uint32_t height) { _die_shape->set_height(height); }
  void set_core_x(int32_t x) { _core_shape->set_x(x); }
  void set_core_y(int32_t y) { _core_shape->set_y(y); }
  void set_core_width(uint32_t width) { _core_shape->set_width(width); }
  void set_core_height(uint32_t height) { _core_shape->set_height(height); }

 private:
  FPRect* _die_shape;
  FPRect* _core_shape;
};

}  // namespace ipl::imp
