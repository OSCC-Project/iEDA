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
#include "ops/read_vcd/VCDParserWrapper.hh"

namespace ipower {

/**
 * @brief The top class of power analysis.
 *
 */
class Power {
 public:
  explicit Power(StaGraph* sta_graph) : _power_graph(sta_graph) {}
  ~Power() = default;

  static Power* getOrCreatePower(StaGraph* sta_graph);
  static void destroyPower();

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

  unsigned buildGraph();
  unsigned readVCD(
      std::string_view vcd_path, std::string top_instance_name,
      std::optional<std::pair<int64_t, int64_t>> begin_end_time = std::nullopt);
  unsigned dumpGraph();
  unsigned buildSeqGraph();
  unsigned dumpSeqGraphViz();

  unsigned setupClock(PwrClock&& fastest_clock, Vector<StaClock*>&& sta_clocks);
  unsigned annotateToggleSP();

  unsigned checkPipelineLoop();
  unsigned levelizeSeqGraph();

  unsigned propagateClock();
  unsigned propagateConst();
  unsigned propagateToggleSP();
  unsigned calcLeakagePower();
  unsigned calcInternalPower();
  unsigned calcSwitchPower();
  unsigned analyzeGroupPower();
  unsigned updatePower();
  unsigned reportPower(const char* rpt_file_name,
                       PwrAnalysisMode pwr_analysis_mode);
  unsigned runCompleteFlow(std::string output_path);

  auto& get_leakage_powers() { return _leakage_powers; }
  auto& get_internal_powers() { return _internal_powers; }
  auto& get_switch_powers() { return _switch_powers; }

  auto& get_type_to_group_data() { return _type_to_group_data; }

 private:
  std::optional<PwrGroupData::PwrGroupType> getInstPowerGroup(
      Instance* the_inst);
  void addGroupData(std::unique_ptr<PwrGroupData> group_data) {
    _type_to_group_data[group_data->get_group_type()].emplace_back(
        group_data.get());
    _group_datas.emplace_back(std::move(group_data));
  }

  PwrGraph _power_graph;          //< The power graph, mapped to sta graph.
  PwrSeqGraph _power_seq_graph;   //!< The power sequential graph, vertex is
                                  //!< sequential inst.
  VcdParserWrapper _vcd_wrapper;  //!< The vcd database.

  std::vector<std::unique_ptr<PwrLeakageData>>
      _leakage_powers;  //!< The leakage power.
  std::vector<std::unique_ptr<PwrInternalData>>
      _internal_powers;  //!< The internal power.
  std::vector<std::unique_ptr<PwrSwitchData>>
      _switch_powers;  //!< The switch power.

  std::vector<std::unique_ptr<PwrGroupData>> _group_datas;  //!< The group data.
  std::map<PwrGroupData::PwrGroupType, std::vector<PwrGroupData*>>
      _type_to_group_data;  //!< The mapping of type to group data.

  static Power* _power;
  DISALLOW_COPY_AND_ASSIGN(Power);
};

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
