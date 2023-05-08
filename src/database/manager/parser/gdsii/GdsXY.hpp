#pragma once

#include <stdint.h>

#include <vector>

namespace idb {

struct XYCoordinate
{
  int32_t x;
  int32_t y;
};

class GdsXY
{
 public:
  GdsXY() = default;
  ~GdsXY() = default;

  // getter
  size_t get_nums() const { return _coords.size(); }
  const std::vector<XYCoordinate>& get_coords() const { return _coords; }
  const XYCoordinate back() const { return _coords.back(); }

  // setter
  void add_coord(int32_t x, int y) { _coords.emplace_back(x, y); }
  void add_coord(const XYCoordinate& c) { _coords.emplace_back(c); }

  // function
  void clear() { _coords.clear(); }

 private:
  std::vector<XYCoordinate> _coords;
};

}  // namespace idb
