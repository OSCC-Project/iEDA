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
 * @file MplHelper.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#ifdef PY_MODEL
#include <Python.h>

#include "PyModel.h"
#endif

#include <type_traits>

#include "python/PyToolBase.hh"
namespace icts {
// Check if type has x and y member variables
template <typename T, typename = std::void_t<>>
struct CoordAble : std::false_type
{
};

template <typename T>
struct CoordAble<T, std::void_t<decltype(T::x), decltype(T::y)>> : std::true_type
{
};

// Check if type can call x() and y() member functions

template <typename T, typename = std::void_t<>>
struct CoordFuncAble : std::false_type
{
};

template <typename T>
struct CoordFuncAble<T, std::void_t<decltype(std::declval<T>().x()), decltype(std::declval<T>().y())>> : std::true_type
{
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

  template <typename PointType>
  typename std::enable_if<CoordAble<PointType>::value>::type plot(const PointType& point, const std::string& label = "")
  {
    pyPlot(_ax, {static_cast<double>(point.x)}, {static_cast<double>(point.y)}, label);
  }
  template <typename PointType>
  typename std::enable_if<CoordFuncAble<PointType>::value>::type plot(const PointType& point, const std::string& label = "")
  {
    pyPlot(_ax, {static_cast<double>(point.x())}, {static_cast<double>(point.y())}, label);
  }

  template <typename PointType>
  typename std::enable_if<CoordAble<PointType>::value>::type plot(const PointType& first, const PointType& second,
                                                                  const std::string& label = "")
  {
    pyPlot(_ax, {static_cast<double>(first.x), static_cast<double>(second.x)},
           {static_cast<double>(first.y), static_cast<double>(second.y)}, label);
  }

  template <typename PointType>
  typename std::enable_if<CoordFuncAble<PointType>::value>::type plot(const PointType& first, const PointType& second,
                                                                      const std::string& label = "")
  {
    pyPlot(_ax, {static_cast<double>(first.x()), static_cast<double>(second.x())},
           {static_cast<double>(first.y()), static_cast<double>(second.y())}, label);
  }

  template <typename PointType>
  typename std::enable_if<CoordAble<PointType>::value>::type plot(const std::vector<PointType>& points, const std::string& label = "")
  {
    std::vector<double> x(points.size());
    std::vector<double> y(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      x[i] = static_cast<double>(points[i].x);
      y[i] = static_cast<double>(points[i].y);
    }

    pyPlot(_ax, x, y, label);
  }

  template <typename PointType>
  typename std::enable_if<CoordFuncAble<PointType>::value>::type plot(const std::vector<PointType>& points, const std::string& label = "")
  {
    std::vector<double> x(points.size());
    std::vector<double> y(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      x[i] = static_cast<double>(points[i].x());
      y[i] = static_cast<double>(points[i].y());
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