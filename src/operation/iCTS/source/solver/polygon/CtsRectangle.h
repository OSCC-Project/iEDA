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

#include "CtsInterval.h"
#include "CtsPoint.h"

namespace icts {

template <typename T>
class CtsRectangle {
 public:
  typedef CtsPoint<T> point_t;
  typedef CtsInterval<T> interval_t;
  typedef T coord_t;

  typedef coord_t coordinate_type;
  typedef interval_t interval_type;

  CtsRectangle() : _ranges() {}
  CtsRectangle(coord_t xl, coord_t yl, coord_t xh, coord_t yh) : _ranges() {
    if (xl > xh) {
      std::swap(xl, xh);
    }
    if (yl > yh) {
      std::swap(yl, yh);
    }
    _ranges[gtl::HORIZONTAL] = interval_t(xl, xh);
    _ranges[gtl::VERTICAL] = interval_t(yl, yh);
  }
  CtsRectangle(const point_t &low, const point_t &high)
      : CtsRectangle(low.x(), low.y(), high.x(), high.y()) {}
  CtsRectangle(point_t low, coord_t width, coord_t height) {
    coord_t xl = low.x();
    coord_t yl = low.y();
    coord_t xh = xl + width;
    coord_t yh = yl + height;
    _ranges[gtl::HORIZONTAL] = interval_t(xl, xh);
    _ranges[gtl::VERTICAL] = interval_t(yl, yh);
  }
  template <typename interval_t_1, typename interval_t_2>
  CtsRectangle(const interval_t_1 &hrange, const interval_t_2 &vrange)
      : _ranges() {
    _ranges[gtl::HORIZONTAL] = hrange;
    _ranges[gtl::VERTICAL] = vrange;
  }
  CtsRectangle(const CtsRectangle &that) : _ranges() { (*this) = that; }

  interval_t get(gtl::orientation_2d orient) const {
    return _ranges[orient.to_int()];
  }
  coord_t get(gtl::direction_2d dir) const {
    return _ranges[gtl::orientation_2d(dir).to_int()].get(
        gtl::direction_1d(dir));
  }
  void set(gtl::direction_2d dir, coord_t value) {
    _ranges[gtl::orientation_2d(dir).to_int()].set(gtl::direction_1d(dir),
                                                   value);
  }

  template <typename interval_t_1>
  void set(gtl::orientation_2d orient, const interval_t_1 &interval) {
    gtl::assign(_ranges[orient.to_int()], interval);
  }

  point_t low() const {
    return point_t(_ranges[gtl::HORIZONTAL].low(),
                   _ranges[gtl::VERTICAL].low());
  }
  point_t high() const {
    return point_t(_ranges[gtl::HORIZONTAL].high(),
                   _ranges[gtl::VERTICAL].high());
  }

  CtsRectangle &operator=(const CtsRectangle &that) {
    if (this != &that) {
      _ranges[0] = that._ranges[0];
      _ranges[1] = that._ranges[1];
    }
    return *this;
  }
  bool operator<(const CtsRectangle &that) const {
    return low() == that.low() ? high() < that.high() : low() < that.low();
  }

  template <typename Rect>
  CtsRectangle &operator=(const Rect &rvalue) {
    gtl::assign(*this, rvalue);
    return *this;
  }
  template <typename Rect>
  bool operator==(const Rect &rvalue) const {
    return gtl::equivalence(*this, rvalue);
  }
  template <typename Rect>
  bool operator!=(const Rect &rvalue) const {
    return !(*this == rvalue);
  }

  bool is_in(const point_t &point) const {
    return gtl::contains(_ranges[0], point.x()) &&
           gtl::contains(_ranges[1], point.y());
  }

 private:
  interval_t _ranges[2];
};

template <typename T>
static inline std::ostream &operator<<(std::ostream &os,
                                       const CtsRectangle<T> &rect) {
  auto min_loc = rect.low();
  auto max_loc = rect.high();

  os << min_loc.x() << " : " << max_loc.y() << std::endl;
  os << max_loc.x() << " : " << max_loc.y() << std::endl;
  os << max_loc.x() << " : " << min_loc.y() << std::endl;
  os << min_loc.x() << " : " << min_loc.y() << std::endl;
  os << min_loc.x() << " : " << max_loc.y();
  return os;
}

}  // namespace icts