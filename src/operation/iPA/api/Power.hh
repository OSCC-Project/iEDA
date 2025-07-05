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
 * @file Power.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The top class of power analysis.
 * @version 0.1
 * @date 2023-01-02
 */

#pragma once

#include "core/PwrAnalysisData.hh"
#include "core/PwrGraph.hh"
#include "core/PwrGroupData.hh"
#include "core/PwrSeqGraph.hh"
#include "include/PwrConfig.hh"
#include "ops/read_vcd/RustVCDParserWrapper.hh"
#include "iIR/api/iIR.hh"
#include "iIR/source/module/power-netlist/PGNetlist.hh"
#include "json/json.hpp"

using namespace iir;

namespace ipower {
/**
 * @brief The top class of power analysis.
 *
 */
class Power {
 public:
  explicit Power(StaGraph* sta_graph) : _power_graph(sta_graph) {
    _ir_analysis.init();
  }
  ~Power() = default;

  static Power* getOrCreatePower(StaGraph* sta_graph);
  static void destroyPower();

  void set_design_work_space(const char* design_work_space) { _design_work_space = design_work_space; }
  const char* get_design_work_space() { return _design_work_space.c_str(); }

  void set_default_toggle(double default_toggle) { _default_toggle = default_toggle; }
  double get_default_toggle() { return _default_toggle; }

  void enableJsonReport() { _is_json_report_enabled = true; }
  bool isJsonReportEnabled() const { return _is_json_report_enabled; }

  auto& get_fastest_clock() { return _power_graph.get_fastest_clock(); }
  void setFastestClock(const char* clock_name, double clock_period_ns) {
    _power_graph.setFastestClock(clock_name, clock_period_ns);
  }

  void setStaClocks(Vector<StaClock*>&& sta_clocks) {
    _power_graph.set_sta_clocks(std::move(sta_clocks));
  }
  auto& getStaClocks() { return _power_graph.get_sta_clocks(); }

  auto& get_power_graph() { return _power_graph; }
  auto& get_power_seq_graph() { return _power_seq_graph; }

  void set_rust_pg_rc_data(const void* rust_pg_rc_data) {
    _rust_pg_rc_data = rust_pg_rc_data;
      // set rc data.
    _ir_analysis.set_rc_data(_rust_pg_rc_data);
  }
  auto* get_rust_pg_rc_data() { return _rust_pg_rc_data; }
  
  auto& get_leakage_powers() { return _leakage_powers; }
  auto& get_internal_powers() { return _internal_powers; }
  auto& get_switch_powers() { return _switch_powers; }
  auto& get_obj_to_datas() { return _obj_to_datas; }
  auto* getObjData(DesignObject* design_obj) {
    return _obj_to_datas.contains(design_obj) ? _obj_to_datas[design_obj].get()
                                              : nullptr;
  }

  auto& get_type_to_group_data() { return _type_to_group_data; }

  std::optional<PwrGroupData::PwrGroupType> getInstPowerGroup(
      Instance* the_inst);
  void addGroupData(std::unique_ptr<PwrGroupData> group_data) {
    _type_to_group_data[group_data->get_group_type()].emplace_back(
        group_data.get());
    _obj_to_datas[group_data->get_obj()] = std::move(group_data);
  }

  unsigned buildGraph();
  unsigned isBuildGraph() { return _power_graph.numVertex() > 0; }
  unsigned readRustVCD(const char* vcd_path, const char* top_instance_name);
  unsigned dumpGraph();
  unsigned buildSeqGraph();
  unsigned dumpSeqGraphViz();

  unsigned setupClock(PwrClock&& fastest_clock, Vector<StaClock*>&& sta_clocks);
  unsigned annotateToggleSP();

  unsigned initPowerGraphData();

  unsigned checkPipelineLoop();
  unsigned levelizeSeqGraph();

  unsigned propagateClock();
  unsigned propagateConst();
  unsigned propagateToggleSP();

  unsigned initToggleSPData();

  unsigned calcLeakagePower();
  unsigned calcInternalPower();
  unsigned calcSwitchPower();
  unsigned analyzeGroupPower();
  unsigned updatePower();

  unsigned reportSummaryPower(const char* rpt_file_name,
                              PwrAnalysisMode pwr_analysis_mode);
  unsigned reportSummaryPowerJSON(const char* rpt_file_name,
                                  PwrAnalysisMode pwr_analysis_mode);
  unsigned reportInstancePower(const char* rpt_file_name,
                               PwrAnalysisMode pwr_analysis_mode);
  unsigned reportInstancePowerCSV(const char* rpt_file_name);

  unsigned reportPower(bool is_copy = true);

  std::vector<IRInstancePower> getInstancePowerData();

  unsigned runCompleteFlow();

  // Below for IR analysis API.
  unsigned readPGSpef(const char* spef_file);
  void resetIRAnalysisData() {
    iIR ir_analysis;
    _ir_analysis = std::move(ir_analysis);
    _rust_pg_rc_data = nullptr;
  }

  double getNominalVoltage() {
    return _ir_analysis.get_nominal_voltage();
  }

  auto& getNetInstanceIRDrop() {
    return _ir_analysis.get_net_to_instance_ir_drop();
  }
  auto getInstanceIRDrop(std::string power_net_name) {
    auto& net_to_instance_ir_drop = _ir_analysis.get_net_to_instance_ir_drop();
    return net_to_instance_ir_drop.at(power_net_name);
  }

  void setBumpNodeLocs(
      const std::map<std::string, IRNodeLoc>& net_bump_node_locs) {
    _ir_analysis.set_net_bump_node_locs(net_bump_node_locs);
  }  
  unsigned runIRAnalysis(std::string power_net_name);
  unsigned reportIRDropTable(const char* rpt_file_name);
  unsigned reportIRDropCSV(const char* rpt_file_name, std::string net_name);
  unsigned reportIRAnalysis(bool is_copy = true);

  std::pair<double, double> getNetToggleAndVoltageData(const char* net_name);

  std::map<ista::Instance::Coordinate, double> displayInstancePowerMap();

 private:
  std::string _design_work_space; //!< The power report work space.
  std::optional<std::pair<std::string, std::string>> _backup_work_dir;
  double _default_toggle = 0.02; //!< The default toggle value.

  PwrGraph _power_graph;         //< The power graph, mapped to sta graph.
  PwrSeqGraph _power_seq_graph;  //!< The power sequential graph, vertex is
                                 //!< sequential inst.
  RustVcdParserWrapper _rust_vcd_wrapper;  //!< The rust vcd database.

  std::vector<std::unique_ptr<PwrLeakageData>>
      _leakage_powers;  //!< The leakage power.
  std::vector<std::unique_ptr<PwrInternalData>>
      _internal_powers;  //!< The internal power.
  std::vector<std::unique_ptr<PwrSwitchData>>
      _switch_powers;  //!< The switch power.

  std::map<DesignObject*, std::unique_ptr<PwrGroupData>> _obj_to_datas;
  std::map<PwrGroupData::PwrGroupType, std::vector<PwrGroupData*>>
      _type_to_group_data;  //!< The mapping of type to group data.

  iIR _ir_analysis; //!< The IR Drop analysis top.
  const void* _rust_pg_rc_data = nullptr; //!< The rust power/ground rc data.

  bool _is_json_report_enabled = false;  //!< Whether to enable json report.

  static Power* _power;
  FORBIDDEN_COPY(Power);
};

/**
 * @brief The macro of foreach group data, usage:
 * Power* ipower;
 * PwrGroupData* group_data;
 * FOREACH_PWR_GROUP_DATA(ipower, group_data)
 * {
 *    do_something_for_group_data();
 * }
 */
#define FOREACH_PWR_GROUP_DATA(ipower, group_data)                            \
  if (auto& group_datas = (ipower)->get_obj_to_datas(); !group_datas.empty()) \
    for (auto p = group_datas.begin();                                        \
         p != group_datas.end() ? group_data = p->second.get(), true : false; \
         ++p)

/**
 * @brief The macro of foreach leakage power, usage:
 * Power* ipower;
 * PwrLeakageData* leakage_power;
 * FOREACH_PWR_LEAKAGE_POWER(ipower, leakage_power)
 * {
 *    do_something_for_leakage_power();
 * }
 */
#define FOREACH_PWR_LEAKAGE_POWER(ipower, leakage_power)                     \
  if (auto& leakage_powers = (ipower)->get_leakage_powers();                 \
      !leakage_powers.empty())                                               \
    for (auto p = leakage_powers.begin();                                    \
         p != leakage_powers.end() ? leakage_power = p->get(), true : false; \
         ++p)

/**
 * @brief The macro of foreach internal power, usage:
 * Power* ipower;
 * PwrInternalData* internal_power;
 * FOREACH_PWR_INTERNAL_POWER(ipower, internal_power)
 * {
 *    do_something_for_internal_power();
 * }
 */
#define FOREACH_PWR_INTERNAL_POWER(ipower, internal_power)                     \
  if (auto& internal_powers = (ipower)->get_internal_powers();                 \
      !internal_powers.empty())                                                \
    for (auto p = internal_powers.begin();                                     \
         p != internal_powers.end() ? internal_power = p->get(), true : false; \
         ++p)

/**
 * @brief The macro of foreach switch  power, usage:
 * Power* ipower;
 * PwrSwitchData* switch_power;
 * FOREACH_PWR_SWITCH_POWER(ipower, switch_power)
 * {
 *    do_something_for_switch_power();
 * }
 */
#define FOREACH_PWR_SWITCH_POWER(ipower, switch_power)                     \
  if (auto& switch_powers = (ipower)->get_switch_powers();                 \
      !switch_powers.empty())                                              \
    for (auto p = switch_powers.begin();                                   \
         p != switch_powers.end() ? switch_power = p->get(), true : false; \
         ++p)

}  // namespace ipower
