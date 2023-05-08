#pragma once

#include <boost/polygon/gtl.hpp>

#include "CtsInterval.h"
#include "CtsPoint.h"

namespace icts {

template <typename T>
class CtsSegment {
 public:
  typedef CtsInterval<T> interval_t;
  typedef CtsPoint<T> point_t;
  typedef T coord_t;
  typedef coord_t coordinate_type;
  typedef point_t point_type;

  CtsSegment() : _points() {}
  CtsSegment(const point_t &low, const point_t &high) {
    _points[gtl::LOW] = low;
    _points[gtl::HIGH] = high;
  }
  CtsSegment(const CtsSegment &that) {
    _points[0] = that._points[0];
    _points[1] = that._points[1];
  }
  CtsSegment &operator=(const CtsSegment &that) {
    _points[0] = that._points[0];
    _points[1] = that._points[1];
    return *this;
  }
  point_t get(gtl::direction_1d dir) const { return _points[dir.to_int()]; }

  point_t low() const { return _points[gtl::LOW]; }
  point_t high() const { return _points[gtl::HIGH]; }

  void set(gtl::direction_1d dir, const point_t &point) {
    _points[dir.to_int()] = point;
  }

  CtsSegment &low(const point_t &point) {
    _points[gtl::LOW] = point;
    return *this;
  }
  CtsSegment &high(const point_t &point) {
    _points[gtl::HIGH] = point;
    return *this;
  }

  bool operator==(const CtsSegment &that) const {
    return _points[0] == that._points[0] && _points[1] == that._points[1];
  }
  bool operator!=(const CtsSegment &that) const { return !(*this == that); }
  bool operator<(const CtsSegment &that) const {
    if (_points[gtl::LOW] != that._points[gtl::LOW]) {
      return _points[gtl::LOW] < that._points[gtl::LOW];
    }
    return _points[gtl::HIGH] < that._points[gtl::HIGH];
  }
  bool operator<=(const CtsSegment &that) const { return !(that < *this); }
  bool operator>(const CtsSegment &that) const { return that < *this; }
  bool operator>=(const CtsSegment &that) const { return !(*this < that); }

 private:
  point_t _points[2];
};

template <typename T>
static inline std::ostream &operator<<(std::ostream &os,
                                       const CtsSegment<T> &segment) {
  os << segment.low() << "\n" << segment.high();
  return os;
}

}  // namespace icts