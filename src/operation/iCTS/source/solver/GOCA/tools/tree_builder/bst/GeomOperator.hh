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
 * @file GeomOperator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <vector>

#include "pgl.h"

namespace icts {
/**
 * @brief geometry operator
 *
 */
namespace bg = boost::geometry;
using bg_Point = boost::geometry::model::d2::point_xy<double>;
using bg_Segment = boost::geometry::model::linestring<bg_Point>;
using bg_Polygon = boost::geometry::model::polygon<bg_Point>;

class GeomOperator
{
 public:
  static Point bgToPglPoint(const bg_Point& p);
  static Segment bgToPglSegment(const bg_Segment& s);
  static Polygon bgToPglPolygon(const bg_Polygon& p);
  static bg_Point pglToBgPoint(const Point& p);
  static bg_Segment pglToBgSegment(const Segment& s);
  static bg_Polygon pglToBgPolygon(const Polygon& p);
  static std::vector<Point> intersectionPointByBg(const Polygon& poly_a, const Polygon& poly_b);
  static std::vector<Point> intersectionPointByBg(const Polygon& poly, const Segment& seg);
  static Point intersectionPointByBg(const Segment& seg_a, const Segment& seg_b);
  static Polygon intersectionByBg(const Polygon& poly_a, const Polygon& poly_b);
  static Polygon intersectionByBg(const Polygon& poly, const Segment& seg);
  static Segment intersectJS(const Segment& js_i, const Segment& js_j, const int& radius_by_j);
  static void calcSDR(Polygon& sdr, const Segment& seg_i, const Segment& seg_j);

  static bool isPoint(const Polygon& poly);
  static bool isSegment(const Polygon& poly);
};
}  // namespace icts