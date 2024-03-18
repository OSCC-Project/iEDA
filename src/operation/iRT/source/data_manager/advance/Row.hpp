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

class Row
{
 public:
  Row() = default;
  ~Row() = default;
  // getter
  int32_t get_staur_x() const { return _staur_x; }
  int32_t get_staur_y() const { return _staur_y; }
  int32_t get_height() const { return _height; }
  // setter
  void set_staur_x(const int32_t staur_x) { _staur_x = staur_x; }
  void set_staur_y(const int32_t staur_y) { _staur_y = staur_y; }
  void set_height(const int32_t height) { _height = height; }
  // function
 private:
  int32_t _staur_x = -1;
  int32_t _staur_y = -1;
  int32_t _height = -1;
};

}  // namespace irt
