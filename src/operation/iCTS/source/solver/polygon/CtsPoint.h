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
#include <boost/polygon/gtl.hpp>
#include <ostream>

namespace icts {

template <typename T>
class CtsPoint
{
 public:
  typedef T coord_t;
  typedef coord_t coordinate_type;

  CtsPoint() = default;
  CtsPoint(coord_t x, coord_t y)
  {
    _coords[gtl::HORIZONTAL] = x;
    _coords[gtl::VERTICAL] = y;
  }
  template <typename PointType>
  CtsPoint(const PointType& that)
  {
    _coords[0] = static_cast<coord_t>(gtl::point_traits<PointType>::get(that, gtl::HORIZONTAL));
    _coords[1] = static_cast<coord_t>(gtl::point_traits<PointType>::get(that, gtl::VERTICAL));
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

  template <typename PointType>
  CtsPoint& operator=(const PointType& that)
  {
    _coords[0] = static_cast<coord_t>(gtl::point_traits<PointType>::get(that, gtl::HORIZONTAL));
    _coords[1] = static_cast<coord_t>(gtl::point_traits<PointType>::get(that, gtl::HORIZONTAL));
    return *this;
  }

  template <typename PointType>
  CtsPoint& operator+=(const PointType& that)
  {
    _coords[0] += gtl::point_traits<PointType>::get(that, gtl::HORIZONTAL);
    _coords[1] += gtl::point_traits<PointType>::get(that, gtl::VERTICAL);
    return *this;
  }

  template <typename PointType>
  CtsPoint& operator-=(const PointType& that)
  {
    _coords[0] -= gtl::point_traits<PointType>::get(that, gtl::HORIZONTAL);
    _coords[1] -= gtl::point_traits<PointType>::get(that, gtl::VERTICAL);
    return *this;
  }

  CtsPoint& operator/=(const coordinate_type& times)
  {
    _coords[0] /= times;
    _coords[1] /= times;
    return *this;
  }

  coord_t get(gtl::orientation_2d orient) const { return _coords[orient.to_int()]; }

  CtsPoint& set(gtl::orientation_2d orient, coord_t value)
  {
    _coords[orient.to_int()] = value;
    return *this;
  }

  coord_t x() const { return _coords[gtl::HORIZONTAL]; }
  coord_t y() const { return _coords[gtl::VERTICAL]; }

  CtsPoint& x(coord_t x)
  {
    _coords[gtl::HORIZONTAL] = x;
    return *this;
  }
  CtsPoint& y(coord_t y)
  {
    _coords[gtl::VERTICAL] = y;
    return *this;
  }

  bool operator==(const CtsPoint& rhs) const { return _coords[0] == rhs._coords[0] && _coords[1] == rhs._coords[1]; }
  bool operator!=(const CtsPoint& that) const { return !(*this == that); }
  bool operator<(const CtsPoint& rhs) const
  {
    return _coords[gtl::VERTICAL] < rhs._coords[gtl::VERTICAL]
           || (_coords[gtl::VERTICAL] == rhs._coords[gtl::VERTICAL] && _coords[gtl::HORIZONTAL] < rhs._coords[gtl::HORIZONTAL]);
  }
  bool operator<=(const CtsPoint& that) const { return !(that < *this); }
  bool operator>(const CtsPoint& that) const { return that < *this; }
  bool operator>=(const CtsPoint& that) const { return !(*this < that); }
  CtsPoint operator-(const CtsPoint& that) const { return CtsPoint(this->x() - that.x(), this->y() - that.y()); }
  CtsPoint operator+(const CtsPoint& that) const { return CtsPoint(this->x() + that.x(), this->y() + that.y()); }
  template <typename TimeType>
  CtsPoint operator/(const TimeType& i) const
  {
    return CtsPoint(static_cast<coord_t>(this->x() / i), static_cast<T>(this->y() / i));
  }
  template <typename TimeType>
  CtsPoint operator*(const TimeType& i) const
  {
    return CtsPoint(static_cast<coord_t>(this->x() * i), static_cast<T>(this->y() * i));
  }

 private:
  coord_t _coords[2];
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const CtsPoint<T>& point)
{
  os << point.x() << " : " << point.y();
  return os;
}

}  // namespace icts
