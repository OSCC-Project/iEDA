// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

#include "api/Power.hh"
#include "sta/Sta.hh"

namespace ipower {

/**
 * @brief interface for python of read vcd.
 *
 * @param vcd_file
 * @param top_instance_name
 * @return true
 * @return false
 */
bool read_vcd(std::string vcd_file, std::string top_instance_name) {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readVCD(vcd_file, top_instance_name);
}

/**
 * @brief interface for python of report power.
 *
 * @return unsigned
 */
unsigned report_power() {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  // set fastest clock for default toggle
  auto* fastest_clock = ista->getFastestClock();
  ipower::PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(),
                                     fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = ista->getClocks();

  std::string output_path = ista->get_design_work_space();
  output_path += Str::printf("/%s.pwr", ista->get_design_name().c_str());

  ipower->setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  ipower->runCompleteFlow(output_path);

  return 1;
}

PYBIND11_MODULE(ipower_cpp, m) {
  m.def("read_vcd_cpp", &read_vcd, py::arg("file_name"), py::arg("top_name"));

  m.def("report_power_cpp", &report_power);
}
}  // namespace ipower