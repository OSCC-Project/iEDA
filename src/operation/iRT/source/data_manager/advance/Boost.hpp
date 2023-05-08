#pragma once

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/polygon.hpp>
#include <iostream>

#include "RTU.hpp"

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BoostPoint = bg::model::d2::point_xy<irt_int, bg::cs::cartesian>;
using BoostBox = bg::model::box<BoostPoint>;
