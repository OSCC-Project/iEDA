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
 * @file CtsPoint.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <ostream>

namespace icts {
constexpr static size_t kX = 0;
constexpr static size_t kY = 1;

template <typename T>
class CtsPoint
{
 public:
  typedef T coord_t;
  typedef coord_t coordinate_type;

  CtsPoint() = default;
  CtsPoint(coord_t x, coord_t y)
  {
    _coords[kX] = x;
    _coords[kY] = y;
  }
  CtsPoint(const CtsPoint& that)
  {
    _coords[0] = that._coords[0];
    _coords[1] = that._coords[1];
  }
  CtsPoint& operator=(const CtsPoint& that)
  {
    _coords[0] = that._coords[0];
    _coords[1] = that._coords[1];
    return *this;
  }

  coord_t x() const { return _coords[kX]; }
  coord_t y() const { return _coords[kY]; }

  CtsPoint& x(const coord_t& x)
  {
    _coords[kX] = x;
    return *this;
  }
  CtsPoint& y(const coord_t& y)
  {
    _coords[kY] = y;
    return *this;
  }

  bool operator==(const CtsPoint& rhs) const { return _coords[kX] == rhs._coords[kX] && _coords[kY] == rhs._coords[kY]; }
  bool operator!=(const CtsPoint& that) const { return !(*this == that); }
  bool operator<(const CtsPoint& rhs) const
  {
    return _coords[kX] < rhs._coords[kX] || (_coords[kX] == rhs._coords[kX] && _coords[kY] < rhs._coords[kY]);
  }
  bool operator<=(const CtsPoint& that) const { return !(that < *this); }
  bool operator>(const CtsPoint& that) const { return that < *this; }
  bool operator>=(const CtsPoint& that) const { return !(*this < that); }
  CtsPoint operator-(const CtsPoint& that) const { return CtsPoint(this->x() - that.x(), this->y() - that.y()); }
  CtsPoint operator+(const CtsPoint& that) const { return CtsPoint(this->x() + that.x(), this->y() + that.y()); }
  template <typename TimeType>
  CtsPoint operator/(const TimeType& i) const
  {
    return CtsPoint(static_cast<coord_t>(this->x() / i), static_cast<coord_t>(this->y() / i));
  }
  template <typename TimeType>
  CtsPoint operator*(const TimeType& i) const
  {
    return CtsPoint(static_cast<coord_t>(this->x() * i), static_cast<coord_t>(this->y() * i));
  }
  CtsPoint operator+=(const CtsPoint& that)
  {
    _coords[0] += that._coords[0];
    _coords[1] += that._coords[1];
    return *this;
  }
  CtsPoint operator-=(const CtsPoint& that)
  {
    _coords[0] -= that._coords[0];
    _coords[1] -= that._coords[1];
    return *this;
  }
  template <typename TimeType>
  CtsPoint operator*=(const TimeType& i)
  {
    _coords[0] *= i;
    _coords[1] *= i;
    return *this;
  }
  template <typename TimeType>
  CtsPoint operator/=(const TimeType& i)
  {
    _coords[0] /= i;
    _coords[1] /= i;
    return *this;
  }

  static bool isRectilinear(const CtsPoint& p1, const CtsPoint& p2) { return p1.x() == p2.x() || p1.y() == p2.y(); }

  static T manhattanDistance(const CtsPoint& p1, const CtsPoint& p2) { return std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y()); }

 private:
  coord_t _coords[2];
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const CtsPoint<T>& point)
{
  os << point.x() << " : " << point.y();
  return os;
}

typedef CtsPoint<int> Point;

}  // namespace icts
