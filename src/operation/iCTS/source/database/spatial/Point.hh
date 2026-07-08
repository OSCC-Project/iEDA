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
/**
 * @file Point.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-07
 * @brief Point class representing a point in 2D space
 */

#pragma once
#include <ostream>

namespace icts {
template <typename T>
class Point
{
 public:
  using CoordType = T;

  Point() = default;

  Point(const CoordType& x, const CoordType& y) : _x(x), _y(y) {}

  Point(const Point&) = default;
  auto operator=(const Point&) -> Point& = default;

  auto get_x() const -> CoordType { return _x; }
  auto get_y() const -> CoordType { return _y; }

  auto set_x(const CoordType& x) -> Point&
  {
    _x = x;
    return *this;
  }

  auto set_y(const CoordType& y) -> Point&
  {
    _y = y;
    return *this;
  }

  auto operator==(const Point& rhs) const -> bool { return _x == rhs._x && _y == rhs._y; }

  auto operator!=(const Point& rhs) const -> bool { return !(*this == rhs); }

  auto operator<(const Point& rhs) const -> bool { return _x < rhs._x || (_x == rhs._x && _y < rhs._y); }

  auto operator<=(const Point& rhs) const -> bool { return !(rhs < *this); }
  auto operator>(const Point& rhs) const -> bool { return rhs < *this; }
  auto operator>=(const Point& rhs) const -> bool { return !(*this < rhs); }

  auto operator+(const Point& rhs) const -> Point { return Point(_x + rhs._x, _y + rhs._y); }

  auto operator-(const Point& rhs) const -> Point { return Point(_x - rhs._x, _y - rhs._y); }

  auto operator+=(const Point& rhs) -> Point&
  {
    _x += rhs._x;
    _y += rhs._y;
    return *this;
  }

  auto operator-=(const Point& rhs) -> Point&
  {
    _x -= rhs._x;
    _y -= rhs._y;
    return *this;
  }

  template <typename Scalar>
  auto operator*(const Scalar& s) const -> Point
  {
    return Point(static_cast<CoordType>(_x * s), static_cast<CoordType>(_y * s));
  }

  template <typename Scalar>
  auto operator/(const Scalar& s) const -> Point
  {
    return Point(static_cast<CoordType>(_x / s), static_cast<CoordType>(_y / s));
  }

  template <typename Scalar>
  auto operator*=(const Scalar& s) -> Point&
  {
    _x = static_cast<CoordType>(_x * s);
    _y = static_cast<CoordType>(_y * s);
    return *this;
  }

  template <typename Scalar>
  auto operator/=(const Scalar& s) -> Point&
  {
    _x = static_cast<CoordType>(_x / s);
    _y = static_cast<CoordType>(_y / s);
    return *this;
  }

 private:
  CoordType _x{};
  CoordType _y{};
};

template <typename T>
inline auto operator<<(std::ostream& os, const Point<T>& p) -> std::ostream&
{
  os << p.get_x() << " : " << p.get_y();
  return os;
}

}  // namespace icts