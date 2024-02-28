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

#include <vector>

namespace ieda_solver {

class EngineGeometry
{
 public:
  EngineGeometry() {}
  virtual ~EngineGeometry() {}
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
  virtual std::vector<std::vector<std::pair<int, int>>> get_polygons_points() = 0;

  virtual bool checkMinArea(int64_t min_area) = 0;
  virtual bool checkOverlap(EngineGeometry* geometry_cmp) = 0;

 private:
};

}  // namespace ieda_solver