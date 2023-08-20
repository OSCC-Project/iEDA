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

#include <boost/polygon/gtl.hpp>

#include "CtsInterval.h"
#include "CtsPoint.h"
#include "CtsPolygon.h"
#include "CtsPolygonSet.h"
#include "CtsRectangle.h"
#include "CtsSegment.h"

namespace icts {

typedef int Coordinate;
typedef CtsPoint<int> Point;
typedef CtsInterval<int> Interval;
typedef CtsSegment<int> Segment;
typedef CtsRectangle<int> Rectangle;
typedef CtsPolygon<int> Polygon;
typedef CtsPolygonSet<int> PolygonSet;

template <typename T>
CtsPoint<double> toDouble(const CtsPoint<T>& point)
{
  return CtsPoint<double>(point.x(), point.y());
}

template <typename T>
CtsSegment<double> toDouble(const CtsSegment<T>& segment)
{
  return CtsSegment<double>(toDouble(segment.source()), toDouble(segment.target()));
}

template <typename T>
CtsRectangle<double> toDouble(const CtsRectangle<T>& rectangle)
{
  return CtsRectangle<double>(toDouble(rectangle.low()), toDouble(rectangle.high()));
}

template <typename T>
CtsPolygon<double> toDouble(const CtsPolygon<T>& polygon)
{
  CtsPolygon<double> double_polygon;
  for (const auto& point : polygon.points()) {
    double_polygon.add_point(toDouble(point));
  }
  return double_polygon;
}
}  // namespace icts

#include "Operator.h"
#include "PolygonRegister.h"