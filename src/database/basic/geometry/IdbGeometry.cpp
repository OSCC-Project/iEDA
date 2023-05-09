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
/**
 * @project		iDB
 * @file		IdbGeometry.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe basic geometry data structure.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbGeometry.h"

#include <algorithm>

namespace idb {

using namespace std;
IdbRect::IdbRect(const int32_t lx, const int32_t ly, const int32_t hx, const int32_t hy, int32_t width)
{
  if (width <= 0) {
    _lx = lx;
    _ly = ly;
    _hx = hx;
    _hy = hy;
  } else {
    /// horizontal
    if (ly == hy) {
      _lx = std::min(lx, hx);
      _ly = ly - (width / 2);
      _hx = std::max(lx, hx);
      _hy = hy + (width / 2);
    }

    /// vertical
    if (lx == hx) {
      _lx = lx - (width / 2);
      _ly = std::min(ly, hy);
      _hx = hx + (width / 2);
      _hy = std::max(ly, hy);
    }
  }
}

void IdbRect::adjustCoordinate(IdbCoordinate<int32_t>* main_point, IdbCoordinate<int32_t>* follow_point, bool adjust_follow)
{
  IdbCoordinate<int32_t> average_coordinate = get_middle_point();
  if (main_point->get_y() == follow_point->get_y()) {
    // horizontal
    main_point->set_y(average_coordinate.get_y());
    if (adjust_follow) {
      follow_point->set_y(average_coordinate.get_y());
    }

  } else if ((main_point->get_x() == follow_point->get_x())) {
    /// vertical
    main_point->set_x(average_coordinate.get_x());
    if (adjust_follow) {
      follow_point->set_x(average_coordinate.get_x());
    }
  } else {
    /// only change the main point coordinate
    main_point->set_x(average_coordinate.get_x());
    main_point->set_y(average_coordinate.get_y());
  }
}

bool IdbRect::isIntersection(IdbRect rect)
{
  if (rect.get_low_x() > get_high_x() || rect.get_high_x() < get_low_x() || rect.get_low_y() > get_high_y()
      || rect.get_high_y() < get_low_y()) {
    return false;
  }

  return true;
}

bool IdbRect::isIntersection(IdbRect* rect)
{
  if (rect->get_low_x() > get_high_x() || rect->get_high_x() < get_low_x() || rect->get_low_y() > get_high_y()
      || rect->get_high_y() < get_low_y()) {
    return false;
  }

  return true;
}

}  // namespace idb
