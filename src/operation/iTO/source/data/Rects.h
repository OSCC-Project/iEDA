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

#include <string>
#include <vector>

namespace ito {

class Rectangle {
 public:
  Rectangle() : _min_x(0), _max_x(0), _min_y(0), _max_y(0){};
  Rectangle(const int lx, const int hx, const int ly, const int hy)
      : _min_x(lx), _max_x(hx), _min_y(ly), _max_y(hy){};
  ~Rectangle() = default;

  int get_x_min() const { return _min_x; }
  int get_x_max() const { return _max_x; }
  int get_y_min() const { return _min_y; }
  int get_y_max() const { return _max_y; }

  int get_dx() { return abs(_max_x - _min_x); }
  int get_dy() { return abs(_max_y - _min_y); }

  /**
   * @brief A point intersects the interior of this rectangle
   *
   * @param x
   * @param y
   * @return true
   * @return false
   */
  bool overlaps(int x, int y) {
    return (x >= _min_x) && (x <= _max_x) && (y >= _min_y) && (y <= _max_y);
  }

 private:
  int _min_x;
  int _max_x;
  int _min_y;
  int _max_y;
};

} // namespace ito
