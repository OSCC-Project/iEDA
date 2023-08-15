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

#include "CtsDBWrapper.h"
#include "GridGraph.h"
#include "pgl.h"

namespace icts {
using namespace idb;

class Placer
{
 public:
  Placer() { init(); }
  Placer(const Placer& placer) = default;
  ~Placer() = default;

  void init();

  // getter
  int get_grid_width() const { return _grid_width; }
  int get_grid_height() const { return _grid_height; }
  int get_spacing_x() const { return _spacing_x; }
  int get_spacing_y() const { return _spacing_y; }

  // setter
  void set_grid_width(int grid_width) { _grid_width = grid_width; }
  void set_grid_height(int grid_height) { _grid_height = grid_height; }
  void set_spacing_x(int spacing_x) { _spacing_x = spacing_x; }
  void set_spacing_y(int spacing_y) { _spacing_y = spacing_y; }

  // select a suitable place to buffer
  void placeInstance(CtsInstance* inst);
  void cancelPlaceInstance(CtsInstance* inst);

  bool isInCore(const Point& location) const;
  Point legalization(const Point& location) const;

 private:
  void placeBlockages();
  Rectangle findPlacedLocation(const Rectangle& rect) const;
  void setBlockage(const Rectangle& rect);
  void resetBlockage(const Rectangle& rect);

  Rectangle coordToGrid(const Rectangle& rect) const;
  Rectangle gridToCoord(const Rectangle& rect) const;
  Point coordToGrid(const Point& locaiton) const;
  Point gridToCoord(const Point& location) const;

 private:
  GridGraph _grid_graph;

  int _spacing_x;
  int _spacing_y;
  int _grid_width;
  int _grid_height;
};
}  // namespace icts