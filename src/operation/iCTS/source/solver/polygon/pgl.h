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

}  // namespace icts

#include "Operator.h"
#include "PolygonRegister.h"