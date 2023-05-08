#pragma once
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <boost/polygon/gtl.hpp>
#include <vector>

#include "CtsPoint.h"
#include "CtsSegment.h"

namespace icts {

template <typename T>
class CtsPolygon {
 public:
  typedef T coord_t;
  typedef CtsPoint<T> point_t;
  typedef CtsSegment<T> segment_t;
  typedef typename std::vector<point_t>::const_iterator iterator_type;

  typedef coord_t coordinate_type;
  typedef point_t point_type;

  CtsPolygon() : _points() {}
  CtsPolygon(const std::vector<point_t> &points) : _points(points) {}
  CtsPolygon(const std::initializer_list<point_t> &points) : _points(points) {}
  template <typename InputIterator>
  CtsPolygon(InputIterator begin, InputIterator end) : _points(begin, end) {}

  template <typename Poly>
  CtsPolygon(const Poly &that) : _points(that.begin(), that.end()) {}
  CtsPolygon(const CtsPolygon &that) : _points(that._points) {}

  CtsPolygon &operator=(const CtsPolygon &that) {
    _points = that._points;
    return *this;
  }
  template <typename Poly>
  CtsPolygon &operator=(const Poly &rvalue) {
    gtl::assign(*this, rvalue);
    return *this;
  }

  template <typename InputIterator>
  CtsPolygon &set(InputIterator begin, InputIterator end) {
    _points.clear();
    _points.insert(_points.end(), begin, end);
    return *this;
  }

  CtsPolygon &set(const std::initializer_list<point_t> &points) {
    _points.clear();
    _points.insert(_points.end(), points.begin(), points.end());
    return *this;
  }

  std::vector<point_t> get_points() const { return _points; }

  std::vector<segment_t> get_edges() const {
    std::vector<segment_t> segments;
    // segments.emplace_back(segment_t(_points.back(), _points.front()));
    if (_points.size() >= 3) {
      for (auto ite = begin() + 1; ite != end(); ++ite) {
        auto &prev = *(ite - 1);
        auto &curr = (*ite);
        segments.emplace_back(segment_t(prev, curr));
      }
    } else {
      segments.emplace_back(segment_t(_points.back(), _points.front()));
    }
    return segments;
  }

  iterator_type begin() const { return _points.begin(); }
  iterator_type end() const { return _points.end(); }
  std::size_t size() const { return _points.size(); }

  bool empty() const { return begin() == end(); }

  bool operator==(const CtsPolygon &that) const {
    if (_points.size() != that._points.size()) {
      return false;
    }
    for (std::size_t i = 0; i < _points.size(); ++i) {
      if (_points[i] != that._points[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const CtsPolygon &that) const { return !(*this == that); }

  void add_point(const point_t &point) { _points.push_back(point); }

 private:
  std::vector<point_t> _points;
};

template <typename T>
static inline std::ostream &operator<<(std::ostream &os,
                                       const CtsPolygon<T> &polygon) {
  for (auto ite = polygon.begin(); ite != polygon.end(); ++ite) {
    os << *ite << "\n";
  }
  if (polygon.begin() != polygon.end()) {
    os << *polygon.begin();
  }
  return os;
}

}  // namespace icts