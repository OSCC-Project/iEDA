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

#include <boost/geometry.hpp>

#include "geometry_boost.h"

namespace ieda_solver {

/**
 * get intersects polygons by compared with geometry_cmp
 */
std::vector<GtlPolygon> GeometryBoost::getIntersects(GeometryBoost* geometry_cmp)
{
  std::vector<GtlPolygon> result_list;
  if (geometry_cmp == nullptr) {
    return result_list;
  }

  auto& polygon_list = get_polygon_list();
  auto& cmp_polygon_list = geometry_cmp->get_polygon_list();

  GtlPolygonSet intersect(_polygon_list & cmp_polygon_list);
  intersect.get(result_list);

  for (auto result : result_list) {
    std::cout << "Check overlap ";
    for (auto pt : result) {
      std::cout << " ( " << pt.x() << " , " << pt.y() << " ) ";
    }

    std::cout << std::endl;
  }

  return result_list;
}
/**
 * return : false = overlap, true = has no overlap
 */
bool GeometryBoost::checkOverlap(EngineGeometry* geometry_cmp)
{
  bool b_result = false;
  /// Cast Operation
  auto intersects = getIntersects(static_cast<GeometryBoost*>(geometry_cmp));
  b_result = intersects.size() > 0 ? false : true;

  return b_result;
}

}  // namespace ieda_solver