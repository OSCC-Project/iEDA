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
#ifndef SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALPOINT_HPP_
#define SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALPOINT_HPP_

#include <iostream>

namespace eval {

template <class T>
class Point
{
 public:
  Point() : _x(-1), _y(-1) {}
  Point(T x, T y) : _x(x), _y(y) {}
  Point(const Point& other)
  {
    _x = other._x;
    _y = other._y;
  }
  Point(Point&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
  }
  Point& operator=(const Point& other)
  {
    _x = other._x;
    _y = other._y;
    return (*this);
  }
  Point& operator=(Point&& other)
  {
    _x = std::move(other._x);
    _y = std::move(other._y);
    return (*this);
  }
  ~Point() = default;

  bool operator==(const Point<T>& other) const { return (_x == other.get_x() && _y == other.get_y()); }
  bool operator!=(const Point& other) { return !((*this) == other); }
  bool operator<(const Point<T>& other) const
  {
    if (get_x() != other.get_x()) {
      return get_x() < other.get_x();
    } else {
      return get_y() < other.get_y();
    }
  }

  T get_x() const { return _x; }
  T get_y() const { return _y; }

  T set_x(const T& x)
  {
    _x = x;
    return _x;
  }

  T set_y(const T& y)
  {
    _y = y;
    return _y;
  }

  T computeDist(const Point<T>& other) const
  {
    T dx = (get_x() > other.get_x()) ? (get_x() - other.get_x()) : (other.get_x() - get_x());
    T dy = (get_y() > other.get_y()) ? (get_y() - other.get_y()) : (other.get_y() - get_y());
    return dx + dy;
  }

  T computeDistX(const Point<T>& other) const
  {
    T dx = (get_x() > other.get_x()) ? (get_x() - other.get_x()) : (other.get_x() - get_x());
    return dx;
  }

  T computeDistY(const Point<T>& other) const
  {
    T dy = (get_y() > other.get_y()) ? (get_y() - other.get_y()) : (other.get_y() - get_y());
    return dy;
  }

  friend std::ostream& operator<<(std::ostream& out, const Point<T>& point)
  {
    out << "[(" << point.get_x() << ", " << point.get_y() << ")]";
    return out;
  }

  bool isUnLegal() const { return _x == -1 && _y == -1; }

 private:
  T _x;
  T _y;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALPOINT_HPP_
