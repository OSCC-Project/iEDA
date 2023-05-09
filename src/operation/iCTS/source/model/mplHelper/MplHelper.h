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

#include "pgl.h"
#include "python/PyToolBase.h"

namespace icts {
class MplHelper : public PyToolBase {
 public:
#ifdef PY_MODEL
  MplHelper() {
    PyToolBase();
    auto* fig_ax = pyGenAX();
    _fig = PyTuple_GET_ITEM(fig_ax, 0);
    _ax = PyTuple_GET_ITEM(fig_ax, 1);
  }

  ~MplHelper() = default;

  void saveFig(const std::string& file_name) { pySaveFig(_fig, file_name); }

  void plot(const icts::Point& point, const std::string& label = "") {
    pyPlotPoint(_ax, point, label);
  }

  void plot(const icts::Segment& segment, const std::string& label = "") {
    pyPlotSegment(_ax, segment, label);
  }

  void plot(const icts::Polygon& polygon, const std::string& label = "") {
    pyPlotPolygon(_ax, polygon, label);
  }

#endif
 private:
#ifdef PY_MODEL
  PyObject* _fig = NULL;
  PyObject* _ax = NULL;
#endif
};
}  // namespace icts