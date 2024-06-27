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

#include "../utility/Utility.h"
namespace ito {
using ito::increaseHash;
class Point
{
 public:
  Point() = default;
  Point(int x, int y) : _x(x), _y(y) {}

  inline bool operator==(const Point& p) const { return (_x == p._x) && (_y == p._y); }

  // 重构加法
  Point operator+(const Point& other) const { return Point(_x + other._x, _y + other._y); }

  // 重构减法
  Point operator-(const Point& other) const { return Point(_x - other._x, _y - other._y); }

  // 重构除法
  template <typename T>
  Point operator/(T divisor) const
  {
    if (divisor == 0) {
      throw std::invalid_argument("Division by zero is not allowed.");
    }
    return Point(static_cast<int>(_x / divisor), static_cast<int>(_y / divisor));
  }

  // 重构乘法
  template <typename T>
  Point operator*(T multiplier) const
  {
    return Point(static_cast<int>(_x * multiplier), static_cast<int>(_y * multiplier));
  }

  friend inline std::ostream& operator<<(std::ostream& os, const Point& pt)
  {
    os << "(" << pt._x << ", " << pt._y << ")";
    return os;
  }

  int get_x() const { return _x; }
  int get_y() const { return _y; }

  static int64_t manhattanDistance(Point p1, Point p2)
  {
    int64_t x0 = p1._x;
    int64_t x1 = p2._x;
    int64_t y0 = p1._y;
    int64_t y1 = p2._y;
    return abs(x0 - x1) + abs(y0 - y1);
  }

  bool isVisit() { return _is_visit; }
  void set_is_visit() { _is_visit = true; }

 private:
  int _x;
  int _y;
  bool _is_visit = false;
};

class PointHash
{
 public:
  size_t operator()(const Point& pt) const
  {
    size_t hash = 5381;
    increaseHash(hash, pt.get_x());
    increaseHash(hash, pt.get_y());
    return hash;
  }
};

class PointEqual
{
 public:
  bool operator()(const Point& pt1, const Point& pt2) const { return pt1.get_x() == pt2.get_x() && pt1.get_y() == pt2.get_y(); }
};
}  // namespace ito