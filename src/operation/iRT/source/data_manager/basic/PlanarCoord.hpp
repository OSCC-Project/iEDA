#pragma once

#include "RTU.hpp"

namespace irt {

class PlanarCoord
{
 public:
  PlanarCoord() = default;
  PlanarCoord(const irt_int x, const irt_int y)
  {
    _x = x;
    _y = y;
  }
  ~PlanarCoord() = default;
  bool operator==(const PlanarCoord& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const PlanarCoord& other) const { return !((*this) == other); }
  // getter
  irt_int get_x() const { return _x; }
  irt_int get_y() const { return _y; }
  // setter
  void set_x(const irt_int x) { _x = x; }
  void set_y(const irt_int y) { _y = y; }
  void set_coord(const irt_int x, const irt_int y)
  {
    _x = x;
    _y = y;
  }
  // function
 private:
  irt_int _x = -1;
  irt_int _y = -1;
};

struct CmpPlanarCoordByXASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const
  {
    return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_y() < b.get_y();
  }
};

struct CmpPlanarCoordByYASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const
  {
    return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_x() < b.get_x();
  }
};

}  // namespace irt
