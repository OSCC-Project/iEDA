/**
 * @file PythonPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief The implemention of python interface.
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "PythonPower.hh"

namespace ipower {
PYBIND11_MODULE(ipower_cpp, m) {
  m.def("read_vcd_cpp", &read_vcd, py::arg("file_name"), py::arg("top_name"));

  m.def("report_power_cpp", &report_power);
}
}  // namespace ipower