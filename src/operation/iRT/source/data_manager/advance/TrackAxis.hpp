// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
