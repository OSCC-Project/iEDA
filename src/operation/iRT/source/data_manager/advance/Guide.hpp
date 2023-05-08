#pragma once

#include "LayerRect.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

namespace irt {

class Guide : public LayerRect
{
 public:
  Guide() = default;
  Guide(const PlanarRect& shape, const irt_int layer_idx) : LayerRect(shape, layer_idx) {}
  Guide(const PlanarRect& shape, const irt_int layer_idx, const PlanarCoord& grid_coord)
      : LayerRect(shape, layer_idx), _grid_coord(grid_coord)
  {
  }
  ~Guide() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid_coord; }
  // setter
  void set_grid_coord(const PlanarCoord& grid_coord) { _grid_coord = grid_coord; }
  void set_grid_coord(const irt_int x, const irt_int y)
  {
    _grid_coord.set_x(x);
    _grid_coord.set_y(y);
  }
  // function

 private:
  PlanarCoord _grid_coord;
};

}  // namespace irt
