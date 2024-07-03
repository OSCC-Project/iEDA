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
#include "EstimateParasitics.h"
#include "Placer.h"
#include "Reporter.h"
#include "SetupOptimizer.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "data_manager.h"
#include "liberty/Lib.hh"
#include "timing_engine.h"

using namespace std;

namespace ito {

bool SetupOptimizer::performGateSizing(float cap_load, float driver_res, Pin* in_pin, Pin* driver_pin)
{
  return heuristicGateSizing(cap_load, driver_res, in_pin, driver_pin);
}

bool SetupOptimizer::heuristicGateSizing(float cap_load, float prev_drive_resis, Pin* in_pin, Pin* out_pin)
{
  auto in_port = in_pin->get_cell_port();
  auto out_port = out_pin->get_cell_port();
  LibCell* driver_lib_cell = out_port->get_ower_cell();
  Vector<LibCell*>* equiv_lib_cells = timingEngine->get_sta_engine()->classifyCells(driver_lib_cell);
  if (!equiv_lib_cells) {
    return false;
  }

  const char* in_port_name = in_port->get_port_name();
  const char* driver_port_name = out_port->get_port_name();

  sort(equiv_lib_cells->begin(), equiv_lib_cells->end(), [=](LibCell* cell1, LibCell* cell2) {
    LibPort* port1 = cell1->get_cell_port_or_port_bus(driver_port_name);
    LibPort* port2 = cell2->get_cell_port_or_port_bus(driver_port_name);
    return (port1->driveResistance() > port2->driveResistance());
  });

  float delay = timingEngine->calcSetupDelayOfGate(cap_load, out_port) + prev_drive_resis * in_port->get_port_cap();

  LibCell* find_repower_size = nullptr;

  for (LibCell* equiv_lib_cell : *equiv_lib_cells) {
    if (strstr(equiv_lib_cell->get_cell_name(), "CLK") != NULL) {
      continue;
    }
    LibPort* eq_driver_port = equiv_lib_cell->get_cell_port_or_port_bus(driver_port_name);
    LibPort* eq_input_port = equiv_lib_cell->get_cell_port_or_port_bus(in_port_name);

    auto gate_delay = timingEngine->calcSetupDelayOfGate(cap_load, eq_driver_port);

    // auto prev_delay = eq_input_port ? prev_drive_resis * eq_input_port->get_port_cap()
    // : prev_drive_resis * in_port->get_port_cap();

    float eq_cell_delay = gate_delay;  // +  prev_delay;

    if (eq_cell_delay < 0.5 * delay) {
      find_repower_size = equiv_lib_cell;
    }
  }

  if (find_repower_size) {
    Instance* instance = out_pin->get_own_instance();
    bool repower_success = timingEngine->repowerInstance(find_repower_size, instance);
    if (repower_success) {
      toDmInst->add_resize_instance_num();
      toEvalInst->invalidNetRC(out_pin->get_net());
      return true;
    }
  }
  return false;
}

}  // namespace ito