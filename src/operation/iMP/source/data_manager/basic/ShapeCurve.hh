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
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WArANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#ifndef IMP_SHAPECURVE_H
#define IMP_SHAPECURVE_H
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "Geometry.hh"
namespace imp {
constexpr double MIN_AR = 1 / 3;
constexpr double MAX_AR = 3.;
// template <typename T>
// class ShapeCurve
// {
//  public:
//   ShapeCurve(T min_area, double min_ar, double max_ar, double inital_ar, double x_inflate, double y_inflate)
//       : _min_area(min_area), _min_ar(min_ar), _max_ar(max_ar), _ar(inital_ar), _x_inflate(x_inflate), _y_inflate(y_inflate)
//   {
//   }
//   ~ShapeCurve() = default;
//   geo::box<T> minimal_box() const
//   {
//     double min_area = static_cast<double>(_min_area);
//     double height = std::ceil(std::sqrt(min_area * _ar));
//     double width = std::ceil(min_area / height);

//     geo::box<double> fbox = geo::scale(geo::make_box(0., 0., width, height), 1. + _x_inflate, 1. + _y_inflate);

//     return geo::make_box(0, 0, static_cast<T>(geo::width(fbox)), static_cast<T>(geo::height(fbox)));
//   }

//  private:
//   double _ar;  // h/w
//   double _min_ar;
//   double _max_ar;
//   double _x_inflate;
//   double _y_inflate;

//   T _min_area;
// };
// template <template <typename> typename Shape, typename T>
// ShapeCurve<T> make_shapecurve(const std::vector<Shape<T>>& shape, double x_inflate = 0., double y_inflate = 0., double inital_ar = 1.,
//                               double min_ar = MIN_AR, double max_ar = MAX_AR)
// {
//   T min_area{0};
//   // T max_width{0};
//   // T max_height{0};
//   for (auto&& i : shape) {
//     min_area += area(i);
//     // max_width = std::max(max_width, width(i));
//     // max_height = std::max(max_height, height(i));
//   }
//   return ShapeCurve<T>(min_area, min_ar, max_ar, inital_ar, x_inflate, y_inflate);
// }

// template <typename T>
// ShapeCurve<T> MergeShapeCurve(const std::vector<ShapeCurve<T>>& shapes)
// {
//   return ShapeCurve<T>();
// }

template <typename T>
class ShapeCurve
{
 public:
  ShapeCurve(double min_ar = 0.33, double max_ar = 3)
      : _min_ar(min_ar), _max_ar(max_ar), _is_discrete(false), _is_continous(false), _is_mixed(false), _width(0), _height(0), _area(0)
  {
  }
  ShapeCurve(const ShapeCurve& other) = default;
  ~ShapeCurve() = default;

  // getter
  bool is_discrete() const { return _is_discrete; }
  bool is_continous() const { return _is_continous; }
  bool is_mixed() const { return _is_mixed; }
  T get_width() const { return _width; }
  T get_height() const { return _height; }
  double get_area() const { return _area; }
  double get_min_ar() const { return _min_ar; }
  double get_max_ar() const { return _max_ar; }
  double get_ar() const { return calAr(_width, _height); }
  const std::vector<std::pair<T, T>>& get_width_list() const { return _width_list; }
  const std::vector<std::pair<T, T>>& get_height_list() const { return _height_list; }

  void setShapes(const std::vector<std::pair<T, T>>& discrete_shapes, float continuous_shapes_area, bool use_clip = true)
  {
    // discrete_shapes does not concern continuous_shape_area
    _width_list.clear();
    _height_list.clear();
    _is_discrete = false;
    _is_continous = false;
    _is_mixed = false;
    // pure continuous shape
    if (discrete_shapes.empty()) {
      setShapesContinous(continuous_shapes_area);
      _is_continous = true;
      return;
    }
    // pure discrete shape
    if (continuous_shapes_area <= 0) {
      setShapesDiscrete(discrete_shapes, use_clip);
      _is_discrete = true;
      return;
    }
    // mixed shapes
    setShapesMixed(discrete_shapes, continuous_shapes_area, use_clip);
    _is_mixed = true;
  }

  bool resizeRandomly(std::uniform_real_distribution<float>& distribution, std::mt19937& generator)
  {
    if (_width_list.empty() || (_width_list.size() == 1 && _width_list[0].first == _width_list[0].second)) {
      return false;  // no other shapes to change
    }
    size_t idx = static_cast<int>(std::floor(distribution(generator) * _width_list.size()));
    auto min_width = _width_list[idx].first;
    auto max_width = _width_list[idx].second;
    _width = min_width + distribution(generator) * (max_width - min_width);
    _area = _width_list[idx].first * _height_list[idx].first;
    _height = _area / _width;
    return true;
  }

 private:
  double _min_ar;  // w / h
  double _max_ar;  // w / h
  bool _is_discrete;
  bool _is_continous;
  bool _is_mixed;
  T _width;
  T _height;
  double _area;  // uses (largest area of discrete_shapes) + (continous_shape_area)
  std::vector<std::pair<T, T>> _width_list;
  std::vector<std::pair<T, T>> _height_list;

  // double _x_inflate;
  // double _y_inflate;
  bool checkShapeAr(const T& width, const T& height) const
  {
    double ar = calAr(width, height);
    return ar >= _min_ar && ar <= _max_ar;
  }
  double calAr(const T& width, const T& height) const { return double(width) / height; }
  T calWidthByAr(double area, double Ar) { return std::sqrt(area * Ar); }
  void setShapesDiscrete(const std::vector<std::pair<T, T>>& discrete_shapes, bool use_clip)
  {
    // discrete_shapes is in ascending order by area
    // force_flag is used by macros, if force_flag == true, not use arr clip
    if (discrete_shapes.empty()) {
      std::cerr << "Error: setting empty shaping curve" << std::endl;
      exit(1);
    }
    // not sorted..

    for (auto& [width, height] : discrete_shapes) {
      // discard odd shapes
      if (use_clip || checkShapeAr(width, height)) {
        _width_list.emplace_back(width, width);
        _height_list.emplace_back(height, height);
      }
    }

    // if no valid shapes left, not use clipping..
    if (_width_list.empty()) {
      for (auto& [width, height] : discrete_shapes) {
        _width_list.emplace_back(width, height);
        _height_list.emplace_back(height, height);
      }
    }

    _width = discrete_shapes[0].first;
    _height = discrete_shapes[0].second;
    _area = discrete_shapes[0].first * discrete_shapes[0].second;
  }

  void setShapesContinous(float continuous_shapes_area)
  {
    if (continuous_shapes_area <= 0) {
      std::cerr << "setting wrong shapes with no discreate shapes and continous_shape_area!" << std::endl;
      exit(1);
    } else {  // pure continuous shape, calculate min_width && max_width based on ar
      _area = continuous_shapes_area;
      T min_width = calWidthByAr(_area, _min_ar);
      T max_width = calWidthByAr(_area, _max_ar);
      _width_list.emplace_back(min_width, max_width);
      _height_list.emplace_back(_area / min_width, _area / max_width);
      // set default shape;
      _width = _width_list[0].first;
      _height = _height_list[0].first;
      return;
    }
  }

  void setShapesMixed(const std::vector<std::pair<T, T>>& discrete_shapes, float continuous_shapes_area, bool use_clip = true)
  {
    T min_width, max_width;
    // _width_list.clear();
    // _height_list.clear();
    auto iter = std::max_element(
        discrete_shapes.begin(), discrete_shapes.end(),
        [](const std::pair<T, T>& p1, const std::pair<T, T>& p2) -> bool { return p1.first * p1.second < p2.first * p2.second; });
    // use the largest area of discrete_shapes
    _area = iter->first * iter->second + continuous_shapes_area;  // discrete-shape-area + continuous-shape-area

    std::vector<std::pair<T, T>> temp_width_list, temp_height_list;
    for (auto& [width, height] : discrete_shapes) {
      min_width = width;
      max_width = _area / height;
      temp_width_list.emplace_back(min_width, max_width);
    }

    // sort width list asending order
    temp_height_list = temp_width_list;
    temp_width_list.clear();
    std::sort(temp_height_list.begin(), temp_height_list.end(),
              [](const std::pair<T, T>& p1, const std::pair<T, T>& p2) -> bool { return p1.first < p2.first; });
    for (auto& interval : temp_height_list) {
      if (temp_width_list.empty() || interval.first > temp_width_list.back().second) {
        temp_width_list.push_back(interval);
      } else if (interval.second > temp_width_list.back().second) {
        temp_width_list.back().second = interval.second;
      }
    }

    temp_height_list.clear();
    for (auto& [width, height] : temp_width_list) {
      temp_height_list.emplace_back(_area / width, _area / height);
    }

    if (!use_clip) {
      _width_list = temp_width_list;
      _height_list = temp_height_list;
    } else {  // if use_clip, clip odd shapes based on ar
      T new_height, new_width;
      for (size_t i = 0; i < temp_width_list.size(); ++i) {
        // 6 possible situations
        if (calAr(temp_width_list[i].first, temp_height_list[i].first) < _min_ar) {
          if (calAr(temp_width_list[i].second, temp_height_list[i].second) <= _min_ar) {
            continue;
          } else {  // clip width-interval by _min_ar
            new_width = calWidthByAr(_area, _min_ar);
            new_height = _area / new_width;
            _width_list.emplace_back(new_width, temp_width_list[i].second);
            _height_list.emplace_back(new_height, temp_height_list[i].second);
            // clip width-interval by _min_ar
            if (calAr(temp_width_list[i].second, temp_height_list[i].second) > _max_ar) {
              auto new_width = calWidthByAr(_area, _max_ar);
              auto new_height = _area / new_width;
              _width_list.back().second = new_width;
              _height_list.back().second = new_height;
              break;  // no valid shapes left
            }
            continue;
          }
        } else if (calAr(temp_width_list[i].second, temp_height_list[i].second) > _max_ar) {
          if (calAr(temp_width_list[i].first, temp_height_list[i].first) >= _max_ar) {
            break;  // no valid shapes left
          } else {
            auto new_width = calWidthByAr(_area, _max_ar);
            auto new_height = _area / new_width;
            _width_list.emplace_back(temp_width_list[i].first, new_width);
            _height_list.emplace_back(temp_height_list[i].first, new_height);
            break;
          }
        } else {  // this interval needn't clip
          _width_list.emplace_back(temp_width_list[i].first, temp_width_list[i].second);
          _height_list.emplace_back(temp_height_list[i].first, temp_height_list[i].second);
        }
      }

      // if no valid shapes after clipping, rollback
      if (_width_list.empty()) {
        _width_list = temp_width_list;
        _height_list = temp_height_list;
      }
    }

    // set default shape
    _width = _width_list[0].first;
    _height = _height_list[0].first;
  }
};

}  // namespace imp

#endif