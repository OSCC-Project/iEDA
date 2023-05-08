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