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
 * check min area in polyset
 */

bool GeometryBoost::checkMinArea(int64_t min_area)
{
  bool b_result = true;

  auto& polygon_list = get_polygon_list();
  for (auto& polygon : polygon_list) {
    int64_t area = gtl::area(polygon);

    if (area < min_area) {
      b_result &= false;

      /// save the violation info
    }
  }

  return b_result;
}

}  // namespace ieda_solver