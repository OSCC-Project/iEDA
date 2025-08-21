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
#include "py_ipw.h"

#include <string>

#include "api/Power.hh"
#include "api/PowerEngine.hh"
#include "sta/Sta.hh"

namespace python_interface {
bool readRustVCD(const char* vcd_path, const char* top_instance_name)
{
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readRustVCD(vcd_path, top_instance_name);
}

/**
 * @brief interface for python of report power.
 *
 * @return unsigned
 */
unsigned report_power()
{
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  ipower->runCompleteFlow();
  return 1;
}

/**
 * @brief interface for python of read pg spef.
 *
 * @param pg_spef_file
 * @return true
 * @return false
 */
bool read_pg_spef(std::string pg_spef_file)
{
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readPGSpef(pg_spef_file.c_str());
}

/**
 * @brief interface for python of report ir drop.
 *
 * @param power_net_name
 * @return unsigned
 */
unsigned report_ir_drop(std::vector<std::string> power_nets)
{
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();

  for (auto power_net_name : power_nets) {
    power_engine->runIRAnalysis(power_net_name);
  }

  power_engine->reportIRAnalysis();

  return 1;
}

unsigned create_data_flow()
{
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
  return power_engine->creatDataflow();
}

std::map<std::size_t, std::vector<ipower::ClusterConnection>> build_connection_map(std::vector<std::set<std::string>> clusters,
                                                                                   std::set<std::string> src_instances, unsigned max_hop)
{
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
  return power_engine->buildConnectionMap(clusters, src_instances, max_hop);
}

std::vector<ipower::MacroConnection> build_macro_connection_map(unsigned max_hop)
{
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
#ifdef USE_GPU
  return power_engine->buildMacroConnectionMapWithGPU(max_hop);
#else
  return power_engine->buildMacroConnectionMap(max_hop);
#endif
}

std::vector<PathWireTimingPowerData> get_wire_timing_power_data(unsigned n_worst_path_per_clock)
{
  auto* ista = ista::Sta::getOrCreateSta();
  auto path_wire_timing_data = ista->reportTimingData(n_worst_path_per_clock);
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
  auto get_net_name = [](const std::string& pin_port_name) {
    auto* ista = ista::Sta::getOrCreateSta();
    auto objs = ista->get_netlist()->findObj(pin_port_name.c_str(), false, false);
    LOG_FATAL_IF(objs.size() != 1);

    auto* pin_or_port = objs[0];
    std::string net_name = pin_or_port->get_net()->get_name();

    return net_name;
  };

  std::vector<PathWireTimingPowerData> ret_timing_data;
  std::string net_name;
  double net_toggle = 0.0;
  double vdd = 0.0;
  for (auto& one_path_wire_timing_data : path_wire_timing_data) {
    PathWireTimingPowerData ret_one_path_data;
    for (auto& wire_timing_data : one_path_wire_timing_data) {
      WireTimingPowerData ret_wire_data;
      ret_wire_data._from_node_name = std::move(wire_timing_data._from_node_name);
      ret_wire_data._to_node_name = std::move(wire_timing_data._to_node_name);
      ret_wire_data._wire_resistance = wire_timing_data._wire_resistance;
      ret_wire_data._wire_capacitance = wire_timing_data._wire_capacitance;
      ret_wire_data._wire_delay = wire_timing_data._wire_delay;
      ret_wire_data._wire_from_slew = wire_timing_data._wire_from_slew;
      ret_wire_data._wire_to_slew = wire_timing_data._wire_to_slew;

      // for power
      std::string& pin_port_name = wire_timing_data._from_node_name;
      auto pin_port_name_vec = Str::split(pin_port_name.c_str(), ":");
      bool is_pin_port = true;
      // judge if the pin_port_name is pin or port.
      if (pin_port_name_vec.size() == 2) {
        pin_port_name = pin_port_name_vec[1];
        if (std::isdigit(pin_port_name[0])) {
          is_pin_port = false;
        }
      }
      
      // update toggle and vdd
      if (is_pin_port) {
        net_name = get_net_name(pin_port_name);
        auto [toggle, voltage] = power_engine->get_power()->getNetToggleAndVoltageData(net_name.c_str());
        net_toggle = toggle;
        vdd = voltage;
      }

      // calculate wire power
      ret_wire_data._wire_power = 0.5 * net_toggle * vdd * wire_timing_data._wire_capacitance;

      ret_one_path_data.emplace_back(std::move(ret_wire_data));
    }

    ret_timing_data.emplace_back(std::move(ret_one_path_data));
  }

  LOG_INFO << "get wire timing power data size: " << ret_timing_data.size();

  return ret_timing_data;
}

}  // namespace python_interface