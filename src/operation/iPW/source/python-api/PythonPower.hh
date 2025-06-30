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

#include "api/Power.hh"
#include "api/PowerEngine.hh"
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

  return ipower->readRustVCD(vcd_file.c_str(), top_instance_name.c_str());
}

/**
 * @brief interface for python of read pg spef.
 * 
 * @param pg_spef_file 
 * @return true 
 * @return false 
 */
bool read_pg_spef(std::string pg_spef_file) {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readPGSpef(pg_spef_file.c_str());
}

/**
 * @brief interface for python of report power.
 *
 * @return unsigned
 */
unsigned report_power() {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  ipower->runCompleteFlow();
  return 1;
}

/**
 * @brief interface for python of report ir drop.
 * 
 * @param power_net_name 
 * @return unsigned 
 */
unsigned report_ir_drop(std::vector<std::string> power_nets) {
  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();

  for (auto& power_net_name : power_nets) {
    power_engine->runIRAnalysis(power_net_name);
  }

  power_engine->reportIRAnalysis();

  return 1;
}

/**
 * @brief display power map.
 * 
 * @return std::map<Instance::Coordinate, double> 
 */
std::map<Instance::Coordinate, double> display_power_map() {
  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();
  auto ret_value = power_engine->displayPowerMap();
  return ret_value;
}

/**
 * @brief display power map.
 * 
 * @return std::map<Instance::Coordinate, double> 
 */
std::map<Instance::Coordinate, double> display_ir_drop_map() {
  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();
  auto ret_value = power_engine->displayIRDropMap();
  return ret_value;
}

// for dataflow.
/**
 * @brief Create a data flow.
 *
 * @return unsigned
 */
unsigned create_data_flow() {
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
  return power_engine->creatDataflow();
}

/**
 * @brief build connection map of dataflow.
 *
 * @param clusters
 * @param max_hop
 * @return std::map<std::size_t, std::vector<ipower::ClusterConnection>>
 */
std::map<std::size_t, std::vector<ipower::ClusterConnection>>
build_connection_map(std::vector<std::set<std::string>> clusters,
                     std::set<std::string> src_instances, unsigned max_hop) {
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
  return power_engine->buildConnectionMap(clusters, src_instances, max_hop);
}

std::vector<MacroConnection> build_macro_connection_map(unsigned max_hop) {
  auto* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
#ifdef USE_GPU
  return power_engine->buildMacroConnectionMapWithGPU(max_hop);
#else
  return power_engine->buildMacroConnectionMap(max_hop);
#endif
}

}  // namespace ipower