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
#include <stdexcept>
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

enum class ShapeType
{
  NONE,
  DISCRETE,
  CONTINOUS,
  MIXED
};

template <typename T>
class ShapeCurve
{
 public:
  ShapeCurve(double min_ar = 0.33, double max_ar = 3)
      : _type(ShapeType::NONE), _min_ar(min_ar), _max_ar(max_ar), _width(0), _height(0), _area(0)
  {
  }
  ShapeCurve(const ShapeCurve& other) = default;
  ~ShapeCurve() = default;

  // getter
  bool is_discrete() const { return _type == ShapeType::DISCRETE; }
  bool is_continous() const { return _type == ShapeType::CONTINOUS; }
  bool is_mixed() const { return _type == ShapeType::MIXED; }
  T get_width() const { return _width; }
  T get_height() const { return _height; }
  double get_area() const { return _area; }
  double get_min_ar() const { return _min_ar; }
  double get_max_ar() const { return _max_ar; }
  double get_ar() const { return calAr(_width, _height); }
  const std::vector<std::pair<T, T>>& get_width_list() const { return _width_list; }
  const std::vector<std::pair<T, T>>& get_height_list() const { return _height_list; }

  void clip(T bound_width, T bound_height)
  {
    // remove shapes larger than given bound
    if (is_discrete()) {
      clipDiscrete(bound_width, bound_height);
    } else if (is_continous()) {
      clipContinous(bound_width, bound_height);
    } else {
      std::cerr << "clip mixed shape Not Implemented.." << std::endl;
      throw std::exception();
    }
  }

  void setShapes(const std::vector<std::pair<T, T>>& discrete_shapes, float continuous_shapes_area, bool ar_clip = true)
  {
    // discrete_shapes does not concern continuous_shape_area
    // pure continuous shape
    if (discrete_shapes.empty()) {
      setShapesContinous(continuous_shapes_area);
    }
    // pure discrete shape
    else if (continuous_shapes_area <= 0) {
      setShapesDiscrete(discrete_shapes, ar_clip);
      return;
    }
    // mixed shapes
    else {
      setShapesMixed(discrete_shapes, continuous_shapes_area, ar_clip);
    }
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
    _area = double(_width_list[idx].first) * _height_list[idx].first;
    _height = _area / _width;
    return true;
  }

  void add_continous_area(double continous_shape_area, bool ar_clip = true)
  {
    // used in fine-shaping
    if (!is_discrete()) {
      throw std::runtime_error("Error, only discrete_shape can add continous area..");
    }
    if (continous_shape_area <= 0) {
      return;
    }
    setShapesMixed(get_discrete_shapes(), continous_shape_area, ar_clip);
  }

  void printShape() const
  {
    assert(_width_list.size() == _height_list.size());
    std::cout << "==============possible_shapes:=================" << std::endl;
    for (size_t i = 0; i < _width_list.size(); ++i) {
      std::cout << "[ " << _width_list[i].first << ", " << _width_list[i].second << " ] ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    for (size_t i = 0; i < _height_list.size(); ++i) {
      std::cout << "[ " << _height_list[i].first << ", " << _height_list[i].second << " ] ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::string type;
    if (is_discrete())
      type = "discrete";
    else if (is_continous())
      type = "continous";
    else if (is_mixed())
      type = "mixed";
    else
      type = "none";
    std::cout << "type: " << type << std::endl;
    std::cout << "width: " << get_width() << std::endl;
    std::cout << "height: " << get_height() << std::endl;
    std::cout << "area: " << get_area() << std::endl;
    std::cout << "min_ar: " << get_min_ar() << std::endl;
    std::cout << "max_ar: " << get_max_ar() << std::endl;
    std::cout << "curr_ar: " << get_ar() << std::endl;
  }

 private:
  double _min_ar;  // w / h
  double _max_ar;  // w / h
  ShapeType _type;
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
  void setShapesDiscrete(const std::vector<std::pair<T, T>>& discrete_shapes, bool ar_clip)
  {
    // discrete_shapes is in ascending order by area
    // force_flag is used by macros, if force_flag == true, not use arr clip
    // _width_list.clear();
    // _height_list.clear();
    clearShape();
    if (discrete_shapes.empty()) {
      throw std::runtime_error("Error: setting empty shaping curve");
    }
    // not sorted..

    for (auto& [width, height] : discrete_shapes) {
      // discard odd shapes
      if (ar_clip || checkShapeAr(width, height)) {
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
    _area = double(discrete_shapes[0].first) * discrete_shapes[0].second;
    _type = ShapeType::DISCRETE;
  }

  void setShapesContinous(float continuous_shapes_area)
  {
    if (continuous_shapes_area <= 0) {
      throw std::runtime_error("setting wrong shapes with no discreate shapes and continous_shape_area!");
    } else {  // pure continuous shape, calculate min_width && max_width based on ar
      // _width_list.clear();
      // _height_list.clear();
      clearShape();
      _area = continuous_shapes_area;
      T min_width = calWidthByAr(_area, _min_ar);
      T max_width = calWidthByAr(_area, _max_ar);
      _width_list.emplace_back(min_width, max_width);
      _height_list.emplace_back(_area / min_width, _area / max_width);
      // set default shape;
      _width = _width_list[0].first;
      _height = _height_list[0].first;
      _type = ShapeType::CONTINOUS;
      return;
    }
  }

  void setShapesMixed(const std::vector<std::pair<T, T>>& discrete_shapes, float continuous_shapes_area, bool ar_clip = true)
  {
    T min_width, max_width;
    // _width_list.clear();
    // _height_list.clear();
    clearShape();
    auto iter = std::max_element(discrete_shapes.begin(), discrete_shapes.end(),
                                 [](const std::pair<T, T>& p1, const std::pair<T, T>& p2) -> bool {
                                   return double(p1.first) * p1.second < double(p2.first) * p2.second;
                                 });
    // use the largest area of discrete_shapes
    _area = double(iter->first) * iter->second + continuous_shapes_area;  // discrete-shape-area + continuous-shape-area

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

    if (!ar_clip) {
      _width_list = temp_width_list;
      _height_list = temp_height_list;
    } else {  // if ar_clip, clip odd shapes based on ar
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

    if (_width_list.size() > 1) {
      _type = ShapeType::MIXED;
    } else {
      _type = ShapeType::CONTINOUS;  // may become continous shape after merging && clipping
    }
  }

  void clipDiscrete(T bound_width, T bound_height)
  {
    if (is_discrete()) {
      return;
    }
    std::vector<std::pair<T, T>> new_width_list;
    std::vector<std::pair<T, T>> new_height_list;
    for (size_t i = 0; i < _width_list.size(); ++i) {
      if (_width_list[i].first <= bound_width && _height_list[i].first <= bound_height) {
        new_width_list.emplace_back(_width_list[i].first, _width_list[i].first);
        new_height_list.emplace_back(_height_list[i].first, _height_list[i].first);
      }
    }
    if (new_width_list.empty()) {
      // std::cerr << "Error, no valid shape left after discrete-shape clip!" << std::endl;
      std::cerr << "Error: no valid shape left" << std::endl;
      throw std::runtime_error("no valid shape left after discrete-shape clip!");
    }
    if (new_width_list.size() < _width_list.size()) {
      _width_list = std::move(new_width_list);
      _height_list = std::move(new_height_list);
      _width = _width_list[0].first;
      _height = _height_list[0].second;
    }
  }

  void clipContinous(T bound_width, T bound_height)
  {
    if (!is_continous()) {
      return;
    }
    auto min_width = _width_list[0].first;
    auto max_width = _width_list[0].second;
    auto min_height = _height_list[0].second;
    auto max_height = _height_list[0].first;
    if (min_width > bound_width || min_height > bound_height) {
      throw std::runtime_error("Error, no valid shape left after continous-shape clip!");
      // std::cerr << "Error, no valid shape left after continous-shape clip!" << std::endl;
    }
    _width_list[0].second = std::min(max_width, bound_width);
    _height_list[0].first = std::min(max_height, bound_height);
    _width_list[0].first = _area / _height_list[0].first;
    _height_list[0].second = _area / _width_list[0].second;
  }

  std::vector<std::pair<T, T>> get_discrete_shapes()
  {
    if (!is_discrete()) {
      throw std::runtime_error("Error, not discrete shape!");
    }
    std::vector<std::pair<T, T>> discrete_shapes;
    for (size_t i = 0; i < _width_list.size(); ++i) {
      discrete_shapes.emplace_back(_width_list[i].first, _height_list[i].first);
    }
    return discrete_shapes;
  }

  void clearShape()
  {
    _width_list.clear();
    _height_list.clear();
    _width = 0;
    _height = 0;
    _area = 0;
    _type = ShapeType::NONE;
  }
};

}  // namespace imp

#endif