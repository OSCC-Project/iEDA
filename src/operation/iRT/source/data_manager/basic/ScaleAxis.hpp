#pragma once

#include "ScaleGrid.hpp"

namespace irt {

class ScaleAxis
{
 public:
  ScaleAxis() = default;
  ~ScaleAxis() = default;
  // getter
  std::vector<ScaleGrid>& get_x_grid_list() { return _x_grid_list; }
  std::vector<ScaleGrid>& get_y_grid_list() { return _y_grid_list; }
  // setter
  void set_x_grid_list(const std::vector<ScaleGrid>& x_grid_list) { _x_grid_list = x_grid_list; }
  void set_y_grid_list(const std::vector<ScaleGrid>& y_grid_list) { _y_grid_list = y_grid_list; }
  // function
 private:
  std::vector<ScaleGrid> _x_grid_list;
  std::vector<ScaleGrid> _y_grid_list;
};
}  // namespace irt
