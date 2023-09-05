/**
 * @file PythonPower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The python api function for ipower.
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

namespace ipower {

bool read_vcd(std::string vcd_file, std::string top_instance_name);
unsigned report_power();

}  // namespace ipower