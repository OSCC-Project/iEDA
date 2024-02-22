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

#include "RTU.hpp"

namespace irt {

class Row
{
 public:
  Row() = default;
  ~Row() = default;
  // getter
  irt_int get_start_x() const { return _start_x; }
  irt_int get_start_y() const { return _start_y; }
  irt_int get_height() const { return _height; }
  // setter
  void set_start_x(const irt_int start_x) { _start_x = start_x; }
  void set_start_y(const irt_int start_y) { _start_y = start_y; }
  void set_height(const irt_int height) { _height = height; }
  // function
 private:
  irt_int _start_x = -1;
  irt_int _start_y = -1;
  irt_int _height = -1;
};

}  // namespace irt
