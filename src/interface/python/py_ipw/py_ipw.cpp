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
#include <string>

#include "api/Power.hh"
#include "sta/Sta.hh"

namespace python_interface {
bool read_vcd(std::string vcd_file, std::string top_instance_name)
{
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readVCD(vcd_file, top_instance_name);
}

unsigned report_power()
{
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  // set fastest clock for default toggle
  auto* fastest_clock = ista->getFastestClock();
  ipower::PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(), fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = ista->getClocks();

  std::string output_path = ista->get_design_work_space();
  output_path += Str::printf("/%s.pwr", ista->get_design_name().c_str());

  ipower->setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  ipower->runCompleteFlow(output_path);

  return 1;
}

}  // namespace python_interface