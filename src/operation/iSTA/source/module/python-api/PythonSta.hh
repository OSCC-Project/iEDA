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
  timing_engine->readDefDesign(def_file, lef_files);
  return 1;
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

  ista->readVerilogWithRustParser(file_name.c_str());
  return true;
}

/**
 * @brief load liberty files.
 *
 * @param lib_files
 * @return true
 * @return false
 */
bool read_liberty(std::vector<std::string>& lib_files) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readLiberty(lib_files);
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
  ista->linkDesignWithRustParser(cell_name.c_str());
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

/**
 * @brief Get the core size object.
 *
 * @return std::pair<int, int>
 */
std::pair<int, int> get_core_size() {
  auto* ista = ista::Sta::getOrCreateSta();
  auto core_size = ista->get_netlist()->get_core_size();
  if (core_size) {
    return {(int)(core_size->_width), (int)(core_size->_height)};
  }

  return {0.0, 0.0};
}

/**
 * @brief get the inst slack of timing map.
 *
 * @param analysis_mode
 * @return std::map<std::pair<double, double>, double>
 */
std::map<std::pair<double, double>, double> display_timing_map(
    const std::string& analysis_mode) {
  auto* ista = ista::Sta::getOrCreateSta();
  if (analysis_mode == "max") {
    return ista->displayTimingMap(AnalysisMode::kMax);
  }

  return ista->displayTimingMap(AnalysisMode::kMin);
}

/**
 * @brief get the inst slack of timing map.
 *
 * @param analysis_mode
 * @return std::map<std::pair<double, double>, double>
 */
std::map<std::pair<double, double>, double> display_timing_tns_map(
    const std::string& analysis_mode) {
  auto* ista = ista::Sta::getOrCreateSta();
  if (analysis_mode == "max") {
    return ista->displayTimingTNSMap(AnalysisMode::kMax);
  }

  return ista->displayTimingTNSMap(AnalysisMode::kMin);
}

/**
 * @brief dispaly the inst slew of map.
 *
 * @param analysis_mode
 * @return std::map<std::pair<double, double>, double>
 */
std::map<std::pair<double, double>, double> display_slew_map(
    const std::string& analysis_mode) {
  auto* ista = ista::Sta::getOrCreateSta();
  if (analysis_mode == "max") {
    return ista->displayTransitionMap(AnalysisMode::kMax);
  }

  return ista->displayTransitionMap(AnalysisMode::kMin);
}

/**
 * @brief Get the used libs in netlist.
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> get_used_libs() {
  auto* ista = ista::Sta::getOrCreateSta();
  auto used_libs = ista->getUsedLibs();

  std::vector<std::string> ret;
  for (auto& lib : used_libs) {
    ret.push_back(lib->get_file_name());
  }

  return ret;
}

/**
 * @brief Only build timing graph.
 *
 */
void build_timing_graph() {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->buildGraph();
}

/**
 * @brief Only update clock timing.
 *
 */
void update_clock_timing() {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->updateClockTiming();
}

/**
 * @brief Print the graph in yaml format.
 *
 * @param graph_file
 */
void dump_graph_data(std::string graph_file) {
  auto* ista = ista::Sta::getOrCreateSta();
  ista->dumpGraphData(graph_file.c_str());
}

/**
 * @brief Get the wire timing data object.
 *
 * @param n_worst_path_per_clock
 * @return std::vector<StaPathWireTimingData>
 */
std::vector<StaPathWireTimingData> get_wire_timing_data(
    unsigned n_worst_path_per_clock) {
  auto* ista = ista::Sta::getOrCreateSta();
  auto path_wire_timing_data = ista->reportTimingData(n_worst_path_per_clock);

  return path_wire_timing_data;
}

}  // namespace ista
