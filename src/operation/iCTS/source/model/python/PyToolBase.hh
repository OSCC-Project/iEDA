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
 * @file PyToolBase.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#ifdef PY_MODEL
#include <stdexcept>

#include "PyModel.h"
#endif

namespace icts {
class PyToolBase
{
 public:
  PyToolBase()
  {
    if (!isInitialized()) {
      PyInit();
      setPyInitialized();
    }
  }

  ~PyToolBase() = default;

 protected:
  void PyInit()
  {
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

  void PyDestroy()
  {
#ifdef PY_MODEL
    Py_Finalize();
#endif
  }

 private:
  static bool& isInitialized()
  {
    static bool py_initialized = false;
    return py_initialized;
  }

  static void setPyInitialized() { isInitialized() = true; }
};
}  // namespace icts