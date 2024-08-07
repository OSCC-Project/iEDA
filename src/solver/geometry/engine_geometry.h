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
#include <stdint.h>

#include <climits>
#include <tuple>
#include <vector>

#include "geometry_point.h"
#include "geometry_polygon.h"
#include "geometry_polygon_set.h"
#include "geometry_rectangle.h"

namespace ieda_solver {

class EngineGeometry
{
 public:
  EngineGeometry() {}
  virtual ~EngineGeometry() {}

  std::tuple<int, int, int, int> bounding_box() { return std::make_tuple(_min_x, _min_y, _max_x, _max_y); }

  /**
   * add rect to engine
   */
  virtual void addRect(int llx, int lly, int urx, int ury) = 0;
  /**
   * get points from polygon list
   * @param
   * std::pair<int, int> : define point x, y
   * std::vector<std::pair<int, int>> : define point list
   * std::vector<std::vector<std::pair<int, int>>> : define polygon list
   */
  //   virtual std::vector<std::vector<std::pair<int, int>>> get_polygons_points() = 0;

  virtual void addGeometry(EngineGeometry* geometry) = 0;
  virtual bool isIntersect(int llx, int lly, int urx, int ury) = 0;

 protected:
  void updateBoundingBox(int llx, int lly, int urx, int ury)
  {
    _min_x = std::min(_min_x, llx);
    _min_y = std::min(_min_y, lly);
    _max_x = std::max(_max_x, urx);
    _max_y = std::max(_max_y, ury);
  }

 private:
  int _min_x = INT_MAX;
  int _min_y = INT_MAX;
  int _max_x = INT_MIN;
  int _max_y = INT_MIN;
};

}  // namespace ieda_solver