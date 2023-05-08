#pragma once

#include "GCellGrid.hpp"

namespace irt {

class GCellAxis
{
 public:
  GCellAxis() = default;
  ~GCellAxis() = default;
  // getter
  std::vector<GCellGrid>& get_x_grid_list() { return _x_grid_list; }
  std::vector<GCellGrid>& get_y_grid_list() { return _y_grid_list; }
  // setter
  void set_x_grid_list(const std::vector<GCellGrid>& x_grid_list) { _x_grid_list = x_grid_list; }
  void set_y_grid_list(const std::vector<GCellGrid>& y_grid_list) { _y_grid_list = y_grid_list; }
  // function
 private:
  std::vector<GCellGrid> _x_grid_list;
  std::vector<GCellGrid> _y_grid_list;
};
}  // namespace irt
