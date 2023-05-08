#ifndef IDRC_SRC_DB_BOOST_TYPE_H_
#define IDRC_SRC_DB_BOOST_TYPE_H_

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/polygon.hpp>

namespace idrc {
class DrcEdge;
class DrcRect;

using namespace boost::polygon::operators;
namespace bp = boost::polygon;
typedef boost::polygon::point_data<int> BoostPoint;
typedef boost::polygon::segment_data<int> BoostSegment;
typedef boost::polygon::polygon_90_data<int> BoostPolygon;
typedef boost::polygon::polygon_90_with_holes_data<int> PolygonWithHoles;
typedef boost::polygon::polygon_90_set_data<int> PolygonSet;
typedef boost::polygon::rectangle_data<int> BoostRect;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<int, bg::cs::cartesian> RTreePoint;
typedef bg::model::segment<RTreePoint> RTreeSegment;
typedef bg::model::box<RTreePoint> RTreeBox;
typedef std::pair<RTreeSegment, DrcEdge*> rtree_edge_value;
typedef std::pair<RTreeBox, DrcRect*> rtree_rect_value;

typedef std::map<std::string, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> LayerNameToRTreeMap;
typedef std::map<int, bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>>> LayerIdToRTreeMap;
typedef bgi::rtree<std::pair<RTreeBox, DrcRect*>, bgi::quadratic<16>> RectRTree;
}  // namespace idrc

#endif