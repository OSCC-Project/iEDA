#pragma once

#ifdef PY_MODEL
#include "PyModel.h"
#endif

namespace icts {
class PyToolBase {
 public:
  PyToolBase() {
    if (!isInitialized()) {
      PyInit();
      setPyInitialized();
    }
  }

  ~PyToolBase() = default;

 protected:
  void PyInit() {
    if (!isInitialized()) {
#ifdef PY_MODEL
      if (PyImport_AppendInittab("PyModel", PyInit_PyModel) == -1) {
        throw std::invalid_argument("Failed to add PyModel to the interpreter");
      }
      Py_Initialize();
      PyImport_ImportModule("PyModel");
#endif
    }
  }

  void PyDestroy() {
#ifdef PY_MODEL
    Py_Finalize();
#endif
  }

 private:
  static bool& isInitialized() {
    static bool py_initialized = false;
    return py_initialized;
  }

  static void setPyInitialized() { isInitialized() = true; }
};
}  // namespace icts