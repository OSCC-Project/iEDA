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
#ifdef PY_MODEL
#include <Python.h>

#include "PyModel.h"
#endif

#include <type_traits>

#include "pgl.h"
#include "python/PyToolBase.h"
namespace icts {
// Check if type has x and y member variables
template <typename T>
struct CoordAble
{
  template <typename U>
  static auto test(int) -> decltype(U::x, U::y, std::true_type());

  template <typename>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
  static constexpr bool value = type::value;
};

// Check if type can call x() and y() member functions
template <typename T>
struct CoordFuncAble
{
  template <typename U>
  static auto test(int) -> decltype(std::declval<U>().x(), std::declval<U>().y(), std::true_type());

  template <typename>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
  static constexpr bool value = type::value;
};

class MplHelper : public PyToolBase
{
 public:
#ifdef PY_MODEL
  MplHelper()
  {
    PyToolBase();
    auto* fig_ax = pyGenAX();
    _fig = PyTuple_GET_ITEM(fig_ax, 0);
    _ax = PyTuple_GET_ITEM(fig_ax, 1);
  }

  ~MplHelper() = default;

  void saveFig(const std::string& file_name) { pySaveFig(_fig, file_name); }

  template <typename T>
  void plot(const icts::CtsPoint<T>& point, const std::string& label = "")
  {
    auto p = toDouble(point);
    pyPlotPoint(_ax, p, label);
  }

  template <typename T>
  void plot(const icts::CtsSegment<T>& segment, const std::string& label = "")
  {
    auto seg = toDouble(segment);
    pyPlotSegment(_ax, seg, label);
  }

  template <typename T>
  void plot(const icts::CtsPolygon<T>& polygon, const std::string& label = "")
  {
    auto poly = toDouble(polygon);
    pyPlotPolygon(_ax, poly, label);
  }

  template <typename PointType>
  void plot(const PointType& point, const std::string& label = "")
  {
    if constexpr (CoordAble<PointType>::value) {
      pyPlot(_ax, {point.x}, {point.y}, label);
    } else if constexpr (CoordFuncAble<PointType>::value) {
      pyPlot(_ax, {point.x()}, {point.y()}, label);
    } else {
      static_assert(std::is_same<PointType, void>::value, "PointType must have x, y members or x(), y() member functions.");
    }
  }

  template <typename PointType>
  void plot(const PointType& first, const PointType& second, const std::string& label = "")
  {
    if constexpr (CoordAble<PointType>::value) {
      pyPlot(_ax, {first.x, second.x}, {first.y, second.y}, label);
    } else if constexpr (CoordFuncAble<PointType>::value) {
      pyPlot(_ax, {first.x(), second.x()}, {first.y(), second.y()}, label);
    } else {
      static_assert(std::is_same<PointType, void>::value, "PointType must have x, y members or x(), y() member functions.");
    }
  }

  template <typename PointType>
  void plot(const std::initializer_list<PointType>& points, const std::string& label = "")
  {
    std::vector<double> x;
    std::vector<double> y;
    if constexpr (CoordAble<PointType>::value) {
      for (const auto& point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
      }
    } else if constexpr (CoordFuncAble<PointType>::value) {
      for (const auto& point : points) {
        x.push_back(point.x());
        y.push_back(point.y());
      }
    } else {
      static_assert(std::is_same<PointType, void>::value, "PointType must have x, y members or x(), y() member functions.");
    }
    pyPlot(_ax, x, y, label);
  }

#endif
 private:
#ifdef PY_MODEL
  PyObject* _fig = NULL;
  PyObject* _ax = NULL;
#endif
};
}  // namespace icts