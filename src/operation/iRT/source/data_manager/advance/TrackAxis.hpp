#pragma once

#include "TrackGrid.hpp"

namespace irt {

class TrackAxis
{
 public:
  TrackAxis() = default;
  ~TrackAxis() = default;
  // getter
  TrackGrid& get_x_track_grid() { return _x_track_grid; }
  TrackGrid& get_y_track_grid() { return _y_track_grid; }
  // setter
  void set_x_track_grid(const TrackGrid& x_track_grid) { _x_track_grid = x_track_grid; }
  void set_y_track_grid(const TrackGrid& y_track_grid) { _y_track_grid = y_track_grid; }
  // function
 private:
  TrackGrid _x_track_grid;
  TrackGrid _y_track_grid;
};
}  // namespace irt
