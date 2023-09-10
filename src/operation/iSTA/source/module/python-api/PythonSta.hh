/**
 * @file PythonSta.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The python api function for iSTA.
 * @version 0.1
 * @date 2023-09-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "sta/Sta.hh"

namespace ista {

/**
 * @brief Set the design workspace object
 *
 * @param design_workspace
 * @return true
 * @return false
 */
bool set_design_workspace(const std::string& design_workspace) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_design_work_space(design_workspace.c_str());
  return true;
}

/**
 * @brief read lef def file for convert netlist.
 *
 * @param lef_files
 * @param def_file
 * @return true
 * @return false
 */
bool read_lef_def(std::vector<std::string>& lef_files,
                  const std::string& def_file) {
  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();

  auto* db_builder = new idb::IdbBuilder();
  db_builder->buildLef(lef_files);

  db_builder->buildDef(def_file);

  auto db_adapter =
      std::make_unique<TimingIDBAdapter>(timing_engine->get_ista());
  db_adapter->set_idb(db_builder);
  unsigned is_ok = db_adapter->convertDBToTimingNetlist();

  return is_ok;
}

/**
 * @brief read netlist file.
 *
 * @param file_name
 * @return true
 * @return false
 */
bool read_netlist(const std::string& file_name) {
  auto* ista = ista::Sta::getOrCreateSta();

  ista->readVerilog(file_name.c_str());
  return true;
}

/**
 * @brief load liberty.
 *
 * @param file_name
 * @return true
 * @return false
 */
bool read_liberty(const std::string& file_name) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readLiberty(file_name.c_str());
  return true;
}

/**
 * @brief link netlist design for flatten.
 *
 * @param cell_name
 * @return true
 * @return false
 */
bool link_design(const std::string& cell_name) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_top_module_name(cell_name.c_str());
  ista->linkDesign(cell_name.c_str());
  return true;
}

/**
 * @brief read spef file.
 *
 * @param file_name
 * @return true
 * @return false
 */
bool read_spef(const std::string& file_name) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readSpef(file_name.c_str());
  return true;
}

/**
 * @brief read sdc file.
 *
 * @param file_name
 * @return true
 * @return false
 */
bool read_sdc(const std::string& file_name) {
  auto* ista = ista::Sta::getOrCreateSta();
  return ista->readSdc(file_name.c_str());
}

/**
 * @brief report timing for analysis.
 *
 * @param digits
 * @param delay_type
 * @param exclude_cell_names
 * @param derate
 * @return true
 * @return false
 */
bool report_timing() {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->buildGraph();
  ista->updateTiming();
  ista->reportTiming({}, true);
  return true;
}

}  // namespace ista
