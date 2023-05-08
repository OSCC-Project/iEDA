#pragma once

#include "PlanarCoord.hpp"

namespace irt {

class EXTPlanarCoord
{
 public:
  EXTPlanarCoord() = default;
  ~EXTPlanarCoord() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid; }
  irt_int get_grid_x() const { return _grid.get_x(); }
  irt_int get_grid_y() const { return _grid.get_y(); }
  PlanarCoord& get_real_coord() { return _real; }
  irt_int get_real_x() const { return _real.get_x(); }
  irt_int get_real_y() const { return _real.get_y(); }
  // const getter
  const PlanarCoord& get_grid_coord() const { return _grid; }
  const PlanarCoord& get_real_coord() const { return _real; }
  // setter
  void set_grid_coord(const PlanarCoord& grid) { _grid = grid; }
  void set_grid_coord(const irt_int x, const irt_int y) { _grid.set_coord(x, y); }
  void set_real_coord(const PlanarCoord& real) { _real = real; }
  void set_real_coord(const irt_int x, const irt_int y) { _real.set_coord(x, y); }
  // function

 private:
  PlanarCoord _grid;
  PlanarCoord _real;
};

}  // namespace irt
