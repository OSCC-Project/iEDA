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
/**
 * @project		iDB
 * @file		IdbDie.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Die Area information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "../IdbObject.h"

namespace bg = boost::geometry;
typedef bg::model::d2::point_xy<int32_t> point_t;
typedef bg::model::polygon<point_t> polygon_t;

namespace idb {

using std::vector;

class IdbDie : public IdbObject
{
 public:
  IdbDie();
  ~IdbDie();

  // getter
  vector<IdbCoordinate<int32_t>*>& get_points() { return _points; }
  uint64_t get_area();
  const int32_t get_llx() { return get_bounding_box()->get_low_x(); }
  const int32_t get_lly() { return get_bounding_box()->get_low_y(); }
  const int32_t get_urx() { return get_bounding_box()->get_high_x(); }
  const int32_t get_ury() { return get_bounding_box()->get_high_y(); }
  const int32_t get_width() { return get_bounding_box()->get_width(); }
  const int32_t get_height() { return get_bounding_box()->get_height(); }

  bool is_polygon() { return _points.size() > RECTANGLE_NUM ? true : false; }
  //   bool is_inside(int32_t x, int32_t y) { return bg::model::contains(_polygon, point_t(x, y)); }

  bool set_bounding_box();

  // operator
  void reset()
  {
    for (auto pt : _points) {
      if (nullptr != pt) {
        delete pt;
        pt = nullptr;
      }
    }

    _points.clear();
  }

  uint32_t add_point(IdbCoordinate<int32_t>* pt);
  uint32_t add_point(int32_t x, int32_t y);

  // verify data
  void print();

 private:
  constexpr static size_t RECTANGLE_NUM = 2;
  /// @brief only support rectangle
  vector<IdbCoordinate<int32_t>*> _points;
  uint64_t _area;
  polygon_t _polygon;
};

}  // namespace idb
