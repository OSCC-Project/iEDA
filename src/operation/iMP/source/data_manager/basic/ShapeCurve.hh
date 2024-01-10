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
#ifndef IMP_SHAPECURVE_H
#define IMP_SHAPECURVE_H
#include <cassert>
#include <vector>

#include "Geometry.hh"
namespace imp {
constexpr double MIN_AR = 1 / 3;
constexpr double MAX_AR = 3.;
template <typename T>
class ShapeCurve
{
 public:
  ShapeCurve(T min_area, double min_ar, double max_ar, double inital_ar, double x_inflate, double y_inflate)
      : _min_area(min_area), _min_ar(min_ar), _max_ar(max_ar), _ar(inital_ar), _x_inflate(x_inflate), _y_inflate(y_inflate)
  {
  }
  ~ShapeCurve() = default;
  geo::box<T> minimal_box() const
  {
    double min_area = static_cast<double>(_min_area);
    double height = std::ceil(std::sqrt(min_area * _ar));
    double width = std::ceil(min_area / height);

    geo::box<double> fbox = geo::scale(geo::make_box(0., 0., width, height), 1. + _x_inflate, 1. + _y_inflate);

    return geo::make_box(0, 0, static_cast<T>(geo::width(fbox)), static_cast<T>(geo::height(fbox)));
  }

 private:
  double _ar;  // h/w
  double _min_ar;
  double _max_ar;
  double _x_inflate;
  double _y_inflate;

  T _min_area;
};
template <template <typename> typename Shape, typename T>
ShapeCurve<T> make_shapecurve(const std::vector<Shape<T>>& shape, double x_inflate = 0., double y_inflate = 0., double inital_ar = 1.,
                              double min_ar = MIN_AR, double max_ar = MAX_AR)
{
  T min_area{0};
  // T max_width{0};
  // T max_height{0};
  for (auto&& i : shape) {
    min_area += area(i);
    // max_width = std::max(max_width, width(i));
    // max_height = std::max(max_height, height(i));
  }
  return ShapeCurve<T>(min_area, min_ar, max_ar, inital_ar, x_inflate, y_inflate);
}

template <typename T>
ShapeCurve<T> MergeShapeCurve(const std::vector<ShapeCurve<T>>& shapes)
{
  return ShapeCurve<T>();
}
}  // namespace imp

#endif