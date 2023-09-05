/**
 * @file PythonReadVcd.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief the python api for read vcd
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "api/Power.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

bool read_vcd(std::string vcd_file, std::string top_instance_name) {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readVCD(vcd_file, top_instance_name);
}

PYBIND11_MODULE(read_vcd_cpp, m) {
  m.def("read_vcd_cpp", &read_vcd, pybind11::arg("file_name"),
        pybind11::arg("top_name"));
}
