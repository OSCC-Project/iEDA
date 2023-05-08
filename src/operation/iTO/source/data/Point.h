#pragma once

#include <string>
#include <vector>

#include "Utility.h"
namespace ito {
using ito::hashIncr;
class Point {
 public:
  Point() = default;
  Point(int x, int y) : _x(x), _y(y) {}

  inline bool operator==(const Point &p) const { return (_x == p._x) && (_y == p._y); }

  int get_x() const { return _x; }
  int get_y() const { return _y; }

  static int64_t manhattanDistance(Point p1, Point p2) {
    int64_t x0 = p1._x;
    int64_t x1 = p2._x;
    int64_t y0 = p1._y;
    int64_t y1 = p2._y;
    return abs(x0 - x1) + abs(y0 - y1);
  }

  bool isVisit() { return _is_visit; }
  void set_is_visit() { _is_visit = true; }

 private:
  int  _x;
  int  _y;
  bool _is_visit = false;
};

class PointHash {
 public:
  size_t operator()(const Point &pt) const {
    size_t hash = 5381;
    hashIncr(hash, pt.get_x());
    hashIncr(hash, pt.get_y());
    return hash;
  }
};

class PointEqual {
 public:
  bool operator()(const Point &pt1, const Point &pt2) const {
    return pt1.get_x() == pt2.get_x() && pt1.get_y() == pt2.get_y();
  }
};
} // namespace ito