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

#include "FPInst.hh"

namespace ipl::imp {

class FPRect
{
 public:
  FPRect(){};
  ~FPRect(){};

  // getter
  int32_t get_x() { return _coordinate->_x; }
  int32_t get_y() { return _coordinate->_y; }
  uint32_t get_width() { return _width; }
  uint32_t get_height() { return _height; }

  // setter
  void set_x(int32_t x) { _coordinate->_x = x; }
  void set_y(int32_t y) { _coordinate->_y = y; }
  void set_width(uint32_t width) { _width = width; }
  void set_height(uint32_t height) { _height = height; }

 private:
  Coordinate* _coordinate = new Coordinate();
  uint32_t _width = 0;
  uint32_t _height = 0;
};

}  // namespace ipl::imp