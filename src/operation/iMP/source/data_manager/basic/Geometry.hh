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
#ifndef IMP_GEOMETRY_H
#define IMP_GEOMETRY_H
#include <boost/geometry.hpp>
#include <boost/geometry/strategies/transform.hpp>
#include <boost/geometry/strategies/transform/matrix_transformers.hpp>
#include <vector>
namespace imp {
namespace geo {
namespace bg = boost::geometry;

/**Geometry*/

template <typename T>
using point = bg::model::d2::point_xy<T>;
template <typename T>
point<T> make_point(T x, T y)
{
  return point<T>(x, y);
}
template <typename T>
using box = bg::model::box<point<T>>;

template <typename T>
box<T> make_box(point<T> p1, point<T> p2)
{
  return box<T>(p1, p2);
}

template <typename T>
box<T> make_box(T x1, T y1, T x2, T y2)
{
  return box<T>(make_point(x1, y1), make_point(x2, y2));
}

template <typename T>
using polygon = bg::model::polygon<point<T>>;
template <typename T>
polygon<T> make_polygon(const std::vector<point<T>>& coordi)
{
  polygon<T> poly;
  bg::assign_points(poly, coordi);
  bg::correct(poly);
  return poly;
}

/**Geometry Information**/

template <template <typename> typename Geometry, typename T>
T lx(const Geometry<T>& geometry)
{
  return lx(bg::return_envelope(geometry));
}

template <template <typename> typename Geometry, typename T>
T ly(const Geometry<T>& geometry)
{
  return ly(bg::return_envelope(geometry));
}

template <typename T>
T lx(const box<T>& b)
{
  return b.min_corner().x();
}

template <typename T>
T ly(const box<T>& b)
{
  return b.min_corner().y();
}

template <template <typename> typename Geometry, typename T>
T height(const Geometry<T>& geometry)
{
  return height(bg::return_envelope(geometry));
}

template <template <typename> typename Geometry, typename T>
T width(const Geometry<T>& geometry)
{
  return width(bg::return_envelope(geometry));
}

template <typename T>
T height(const box<T>& b)
{
  return b.max_corner().y() - b.min_corner().y();
}

template <typename T>
T width(const box<T>& b)
{
  return b.max_corner().x() - b.min_corner().x();
}

template <typename Geometry>
auto area(const Geometry& geometry)
{
  return bg::area(geometry);
}

/**Geometry Transform**/
/**Transformation always keeps the min corner of boundingbox of geometry unchanged*/
/**
 * TODO: There are a lot better way to transform such as seft-define transform matrix need develop in the future.
 */

template <typename Geometry>
Geometry translate(const Geometry& geometry, decltype(lx(geometry)) x_translate, decltype(ly(geometry)) y_translate)
{
  using namespace boost::geometry::strategy::transform;
  translate_transformer<decltype(lx(geometry)), 2, 2> translate(x_translate, y_translate);
  Geometry shape2;
  bg::transform(geometry, shape2, translate);
  return shape2;
}
/**
 * @brief Rotate counterclockwise around the lower left corner, then move back to the lower left corner.
 *
 *      ____
 *     |   \|
 *     |    |
 *     |    |
 *     .____|
 *  (lx,ly)
 * rotate by 90 degree.
 *      _________
 *     |/        |
 *     ._________|
 *  (lx,ly)
 *
 * @tparam Geometry
 * @param geometry
 * @param degree
 * @return Geometry
 */
template <typename Geometry>
Geometry rotate(const Geometry& geometry, double degree)
{
  using namespace boost::geometry::strategy::transform;
  auto x = lx(geometry);
  auto y = ly(geometry);
  Geometry shape2 = translate(geometry, -x, -y);
  rotate_transformer<bg::degree, double, 2, 2> rotate(-degree);
  bg::transform(shape2, shape2, rotate);
  return translate(shape2, -lx(shape2) + x, -ly(shape2) + y);
}

template <typename Geometry>
Geometry scale(const Geometry& geometry, double x_scale, double y_scale)
{
  using namespace boost::geometry::strategy::transform;
  auto x = lx(geometry);
  auto y = ly(geometry);
  Geometry shape2 = translate(geometry, -x, -y);
  scale_transformer<double, 2, 2> scale(x_scale, y_scale);
  bg::transform(shape2, shape2, scale);
  return translate(shape2, -lx(shape2) + x, -ly(shape2) + y);
}

template <typename Geometry>
Geometry transform(const Geometry& geometry, double x_scale, double y_scale, double degree, decltype(lx(geometry)) x_translate,
                   decltype(ly(geometry)) y_translate)
{
  return translate(rotate(scale(geometry, x_scale, y_scale), degree), x_translate, y_translate);
}

/**Geometry Algorithm**/
template <typename Geometry1, typename Geometry2>
bool within(const Geometry1& geometry1, const Geometry2& geometry2)
{
  return bg::within(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
bool overlaps(const Geometry1& geometry1, const Geometry2& geometry2)
{
  return bg::overlaps(geometry1, geometry2);
}
template <typename Geometry1, typename Geometry2, typename GeometryOut>
bool intersection(const Geometry1& geometry1, const Geometry2& geometry2, GeometryOut& output)
{
  return bg::intersection(geometry1, geometry2, output);
}
// template <typename Geometry1, typename Geometry2>
// std::vector<polygon<decltype(lx(geometry1))>> return_intersection(const Geometry1& geometry1, const Geometry2& geometry2 )
// {
//   std::vector<polygon<decltype(lx(Geometry1))>> output;
//   bg::intersection(geometry1, geometry2, output);
//   return output;
// }
}  // namespace geo
}  // namespace imp
#endif