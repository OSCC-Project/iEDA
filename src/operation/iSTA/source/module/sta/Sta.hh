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
 * @file Sta.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The top interface class of static timing analysis.
 * @version 0.1
 * @date 2020-11-27
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "HashMap.hh"
#include "StaClock.hh"
#include "StaClockTree.hh"
#include "StaGraph.hh"
#include "StaPathData.hh"
#include "StaReport.hh"
#include "Type.hh"
#include "aocv/AocvParser.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "liberty/Liberty.hh"
#include "netlist/Netlist.hh"
#include "parser/liberty/mLibertyEquivCells.hh"
#include "sdc/SdcSetIODelay.hh"
#include "verilog/VerilogReader.hh"

namespace ista {

class SdcConstrain;

constexpr int g_global_derate_num = 8;

// minHeap of the StaSeqPathData.
const std::function<bool(StaSeqPathData*, StaSeqPathData*)> cmp =
    [](StaSeqPathData* left, StaSeqPathData* right) -> bool {
  unsigned left_slack = left->getSlack();
  unsigned right_slack = right->getSlack();
  return left_slack > right_slack;
};

/**
 * @brief The derate table used for delay calc.
 *
 */
class StaDreateTable {
 public:
  enum class DerateIndex : int {
    kMaxClockCell = 0,
    kMaxClockNet = 1,
    kMaxDataCell = 2,
    kMaxDataNet = 3,
    kMinClockCell = 4,
    kMinClockNet = 5,
    kMinDataCell = 6,
    kMinDataNet = 7
  };

  void init() {
    std::for_each(_global_derate_table.begin(), _global_derate_table.end(),
                  [](auto& elem) { elem = 1.0; });
  }

  void set_global_derate_table(int index, double derate_value) {
    _global_derate_table[index] = derate_value;
  }
  auto& operator[](int index) { return _global_derate_table[index]; }

  auto& getMaxClockCellDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMaxClockCell)];
  }
  auto& getMaxClockNetDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMaxClockNet)];
  }

  auto& getMaxDataCellDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMaxDataCell)];
  }
  auto& getMaxDataNetDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMaxDataNet)];
  }

  auto& getMinClockCellDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMinClockCell)];
  }
  auto& getMinClockNetDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMinClockNet)];
  }

  auto& getMinDataCellDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMinDataCell)];
  }
  auto& getMinDataNetDerate() {
    return _global_derate_table[static_cast<int>(DerateIndex::kMinDataNet)];
  }

 private:
  std::array<std::optional<double>, g_global_derate_num> _global_derate_table;
};

/**
 * @brief The report specific to from/through/to.
 *
 */
class StaReportSpec {
 public:
  using ReportList = std::vector<std::string>;
  void set_prop_froms(ReportList&& prop_froms) {
    _prop_froms = std::move(prop_froms);
  }
  auto& get_prop_froms() { return _prop_froms; }

  void set_prop_tos(ReportList&& prop_tos) { _prop_tos = std::move(prop_tos); }
  auto& get_prop_tos() { return _prop_tos; }

  void set_prop_throughs(std::vector<ReportList>&& prop_throughs) {
    _prop_throughs = std::move(prop_throughs);
  }
  auto& get_prop_throughs() { return _prop_throughs; }

 private:
  ReportList _prop_froms;
  ReportList _prop_tos;
  std::vector<ReportList> _prop_throughs;
};

/**
 * @brief The top Sta class, would provide the API to other tools.
 *
 */
class Sta {
 public:
  enum RiseFall : int { kRise = 0, kFall = 1 };

  static Sta* getOrCreateSta();
  static void destroySta();

  void set_design_work_space(const char* design_work_space);
  const char* get_design_work_space() { return _design_work_space.c_str(); }

  void set_num_threads(unsigned num_thread) { _num_threads = num_thread; }
  [[nodiscard]] unsigned get_num_threads() const { return _num_threads; }

  void set_n_worst_path_per_clock(unsigned n_worst) {
    _n_worst_path_per_clock = n_worst;
  }
  [[nodiscard]] unsigned get_n_worst_path_per_clock() const {
    return _n_worst_path_per_clock;
  }

  void set_n_worst_path_per_endpoint(unsigned n_worst) {
    _n_worst_path_per_endpoint = n_worst;
  }
  [[nodiscard]] unsigned get_n_worst_path_per_endpoint() const {
    return _n_worst_path_per_endpoint;
  }

  void set_path_group(std::string&& path_group) {
    _path_group = std::move(path_group);
  }
  auto& get_path_group() { return _path_group; }

  auto& get_clock_groups() const { return _clock_groups; }

  // void initScriptEngine();
  SdcConstrain* getConstrain();

  unsigned readDesign(const char* verilog_file);
  unsigned readLiberty(const char* lib_file);
  unsigned readLiberty(std::vector<std::string>& lib_files);
  unsigned readSdc(const char* sdc_file);
  unsigned readSpef(const char* spef_file);
  unsigned readAocv(const char* aocv_file);
  unsigned readAocv(std::vector<std::string>& aocv_files);

  VerilogModule* findModule(const char* module_name);
  void set_verilog_modules(
      std::vector<std::unique_ptr<VerilogModule>>&& verilog_modules) {
    _verilog_modules = std::move(verilog_modules);
  }

  void set_top_module_name(const char* top_module_name) {
    _top_module_name = top_module_name;
  }
  auto& get_top_module_name() { return _top_module_name; }

  void readVerilog(const char* verilog_file);
  void linkDesign(const char* top_cell_name);
  void set_design_name(const char* design_name) {
    _netlist.set_name(design_name);
  }
  std::string get_design_name() { return _netlist.get_name(); }

  auto& get_constrains() { return _constrains; }
  void resetConstraint();

  Netlist* get_netlist() { return &_netlist; }
  void resetNetlist() { _netlist.reset(); }

  void addLib(std::unique_ptr<LibertyLibrary> lib) {
    std::unique_lock<std::mutex> lk(_mt);
    _libs.emplace_back(std::move(lib));
  }

  LibertyLibrary* getOneLib() {
    return _libs.empty() ? nullptr : _libs.back().get();
  }

  Vector<std::unique_ptr<LibertyLibrary>>& getAllLib() { return _libs; }

  void resetRcNet(Net* the_net) {
    if (_net_to_rc_net.contains(the_net)) {
      _net_to_rc_net.erase(the_net);
    }
  }

  void addRcNet(Net* the_net, std::unique_ptr<RcNet> rc_net) {
    _net_to_rc_net[the_net] = std::move(rc_net);
  }
  void removeRcNet(Net* the_net) { _net_to_rc_net.erase(the_net); }
  RcNet* getRcNet(Net* the_net) {
    return _net_to_rc_net.contains(the_net) ? _net_to_rc_net[the_net].get()
                                            : nullptr;
  }
  void resetAllRcNet() { _net_to_rc_net.clear(); }

  LibertyCell* findLibertyCell(const char* cell_name);
  std::optional<AocvObjectSpecSet*> findDataAocvObjectSpecSet(
      const char* object_name);
  std::optional<AocvObjectSpecSet*> findClockAocvObjectSpecSet(
      const char* object_name);
  void makeEquivCells(std::vector<LibertyLibrary*>& equiv_libs,
                      std::vector<LibertyLibrary*>& map_libs);

  Vector<LibertyCell*>* equivCells(LibertyCell* cell);

  static void initSdcCmd();

  void addClock(std::unique_ptr<StaClock>&& clock) {
    _clocks.emplace_back(std::move(clock));
  }
  Vector<std::unique_ptr<StaClock>>& get_clocks() { return _clocks; }
  Vector<StaClock*> getClocks() {
    Vector<StaClock*> clocks;
    for (auto& clock : _clocks) {
      clocks.emplace_back(clock.get());
    }
    return clocks;
  }

  void clearClocks() { _clocks.clear(); }

  StaClock* findClock(const char* clock_name);
  StaClock* getFastestClock();

  void setIdealClockNetworkLatency(double latency);
  void setIdealClockNetworkLatency(const char* clock_name, double latency);

  void addIODelayConstrain(StaVertex* port_vertex,
                           SdcSetIODelay* io_delay_constrain);
  std::list<SdcSetIODelay*> getIODelayConstrain(StaVertex* port_vertex);
  void clearIODelayConstrain() { _io_delays.clear(); }

  void resetSdcConstrain() {
    clearClocks();
    clearIODelayConstrain();
  }

  void set_analysis_mode(AnalysisMode analysis_mode) {
    _analysis_mode = analysis_mode;
  }
  AnalysisMode get_analysis_mode() { return _analysis_mode; }

  void set_derate_table(const StaDreateTable& dereate_table) {
    _derate_table = dereate_table;
  }
  auto& get_derate_table() { return _derate_table; }

  void addAocv(std::unique_ptr<AocvLibrary> aocv) {
    std::unique_lock<std::mutex> lk(_mt);
    _aocvs.emplace_back(std::move(aocv));
  }
  AocvLibrary* getOneAocv() {
    return _aocvs.empty() ? nullptr : _aocvs.back().get();
  }
  Vector<std::unique_ptr<AocvLibrary>>& getAllAocv() { return _aocvs; }

  void setMaxRiseCap(double cap) { _max_cap[kRise] = cap; }
  auto& getMaxRiseCap() { return _max_cap[kRise]; }
  void setMaxFallCap(double cap) { _max_cap[kFall] = cap; }
  auto& getMaxFallCap() { return _max_cap[kFall]; }

  void setMaxRiseSlew(double slew) { _max_slew[kRise] = slew; }
  auto& getMaxRiseSlew() { return _max_slew[kRise]; }
  void setMaxFallSlew(double slew) { _max_slew[kFall] = slew; }
  auto& getMaxFallSlew() { return _max_slew[kFall]; }
  std::optional<double> getVertexSlewLimit(StaVertex* the_vertex,
                                           AnalysisMode mode,
                                           TransType trans_type);
  std::optional<double> getVertexSlewSlack(StaVertex* the_vertex,
                                           AnalysisMode mode,
                                           TransType trans_type);

  std::optional<double> getVertexCapacitanceLimit(StaVertex* the_vertex,
                                                  AnalysisMode mode,
                                                  TransType trans_type);
  std::optional<double> getVertexCapacitanceSlack(StaVertex* the_vertex,
                                                  AnalysisMode mode,
                                                  TransType trans_type);

  double getVertexCapacitance(StaVertex* the_vertex, AnalysisMode mode,
                              TransType trans_type);

  std::optional<double> getDriverVertexFanoutLimit(StaVertex* the_vertex,
                                                   AnalysisMode mode);
  std::optional<double> getDriverVertexFanoutSlack(StaVertex* the_vertex,
                                                   AnalysisMode mode);

  void setMaxFanout(double max_fanout) { _max_fanout = max_fanout; }
  auto& getMaxFanout() { return _max_fanout; }

  unsigned buildGraph();
  void resetGraph() { _graph.reset(); }
  StaGraph& get_graph() { return _graph; }
  bool isBuildGraph() { return !_graph.get_vertexes().empty(); }

  StaVertex* findVertex(const char* pin_name);
  StaVertex* findVertex(DesignObject* obj) {
    auto the_vertex = _graph.findVertex(obj);
    return the_vertex ? *the_vertex : nullptr;
  }

  bool isMaxAnalysis() {
    return (_analysis_mode == AnalysisMode::kMax ||
            _analysis_mode == AnalysisMode::kMaxMin);
  }
  bool isMinAnalysis() {
    return (_analysis_mode == AnalysisMode::kMin ||
            _analysis_mode == AnalysisMode::kMaxMin);
  }

  unsigned insertPathData(StaClock* capture_clock, StaVertex* end_vertex,
                          StaSeqPathData* seq_data);
  unsigned insertPathData(StaVertex* end_vertex,
                          StaClockGatePathData* seq_data);

  std::unique_ptr<StaReportTable>& get_report_tbl_summary() {
    return _report_tbl_summary;
  }

  void set_significant_digits(unsigned significant_digits) {
    _significant_digits = significant_digits;
  }
  [[nodiscard]] unsigned get_significant_digits() const {
    return _significant_digits;
  }

  void setReportSpec(std::vector<std::string>&& prop_froms,
                     std::vector<std::string>&& prop_tos);

  void setReportSpec(std::vector<std::string>&& prop_froms,
                     std::vector<StaReportSpec::ReportList>&& prop_throughs,
                     std::vector<std::string>&& prop_tos);

  void set_report_spec(StaReportSpec&& report_spec) {
    _report_spec = std::move(report_spec);
  }
  auto& get_report_spec() { return _report_spec; }

  unsigned reportPath(const char* rpt_file_name, bool is_derate = true);
  unsigned reportTrans(const char* rpt_file_name);
  unsigned reportCap(const char* rpt_file_name, bool is_clock_cap);
  unsigned reportFanout(const char* rpt_file_name);
  unsigned reportSkew(const char* rpt_file_name, AnalysisMode analysis_mode);
  unsigned reportFromThroughTo(const char* rpt_file_name,
                               AnalysisMode analysis_mode, const char* from_pin,
                               const char* through_pin, const char* to_pin);

  void reset_clock_groups() {
    _clock_groups.clear();
    _clock_gate_group.reset(nullptr);
  }
  void resetReportTbl() {
    _report_tbl_summary = StaReportPathSummary::createReportTable("sta");
    _report_tbl_TNS = StaReportClockTNS::createReportTable("TNS");
    _report_tbl_details.clear();
  }

  auto& get_report_tbl_TNS() { return _report_tbl_TNS; }
  auto& get_report_tbl_details() { return _report_tbl_details; }
  auto& get_clock_trees() { return _clock_trees; }
  void addClockTree(StaClockTree* clock_tree) {
    _clock_trees.emplace_back(clock_tree);
  }

  StaSeqPathData* getSeqData(StaVertex* vertex, StaData* delay_data);
  double getWNS(const char* clock_name, AnalysisMode mode);
  double getTNS(const char* clock_name, AnalysisMode mode);
  double getLocalSkew(const char* clock_name, AnalysisMode mode,
                      TransType trans_type);
  double getGlobalSkew(AnalysisMode mode, TransType trans_type);
  std::map<StaVertex*, int> getFFMaxSkew(AnalysisMode mode,
                                         TransType trans_type);
  std::map<StaVertex*, int> getFFTotalSkew(AnalysisMode mode,
                                           TransType trans_type);
  std::multimap<std::string, std::string> getSkewRelatedSink(
      AnalysisMode mode, TransType trans_type);
  StaSeqPathData* getWorstSeqData(std::optional<StaVertex*> vertex,
                                  AnalysisMode mode, TransType trans_type);

  StaSeqPathData* getWorstSeqData(AnalysisMode mode, TransType trans_type);

  std::priority_queue<StaSeqPathData*, std::vector<StaSeqPathData*>,
                      decltype(cmp)>
  getViolatedSeqPathsBetweenTwoSinks(StaVertex* vertex1, StaVertex* vertex2,
                                     AnalysisMode mode);
  std::optional<double> getWorstSlackBetweenTwoSinks(StaVertex* vertex1,
                                                     StaVertex* vertex2,
                                                     AnalysisMode mode);
  std::map<std::pair<StaVertex*, StaVertex*>, double>
  getWorstSlackBetweenTwoSinks(AnalysisMode mode);
  int getWorstSlack(StaVertex* end_vertex, AnalysisMode mode,
                    TransType trans_type);
  void writeVerilog(const char* verilog_file_name,
                    std::set<std::string>& exclude_cell_names);

  unsigned resetGraphData();
  unsigned resetPathData();
  unsigned updateTiming();
  unsigned updateClockTiming();
  std::set<std::string> findStartOrEnd(StaVertex* the_vertex, bool is_find_end);
  unsigned reportTiming(std::set<std::string>&& exclude_cell_names = {},
                        bool is_derate = true, bool is_clock_cap = false);

  void dumpVertexData(std::vector<std::string> vertex_names);
  void dumpNetlistData();
  void buildClockTrees();
  void buildNextPin(
      StaClockTree* clock_tree, StaClockTreeNode* parent_node,
      StaVertex* parent_vertex,
      std::map<StaVertex*, std::vector<StaData*>>& vertex_to_datas);

 private:
  Sta();
  ~Sta();

  std::string _design_work_space;

  unsigned _num_threads = 48;  //!< The num of thread for propagation.
  unsigned _n_worst_path_per_clock =
      3;  //!< The top n worst path config for each clock.
  unsigned _n_worst_path_per_endpoint = 1;    //!< The top n worst path
                                              //!< config for each endpoint.
  std::optional<std::string> _path_group;     //!< The path group.
  std::unique_ptr<SdcConstrain> _constrains;  //!< The sdc constrain.
  VerilogReader _verilog_reader;
  std::string _top_module_name;
  std::vector<std::unique_ptr<VerilogModule>>
      _verilog_modules;  //!< The current design parsed from verilog file.
  VerilogModule* _top_module = nullptr;  //!< The design top module.
  Netlist _netlist;  //!< The current top netlist for sta analysis.
  Vector<std::unique_ptr<LibertyLibrary>>
      _libs;  //!< The design libs of different corners.

  std::unique_ptr<LibertyEquivCells>
      _equiv_cells;  //!< The function equivalently liberty cell.

  AnalysisMode _analysis_mode;  //!< The analysis max/min mode.

  StaDreateTable _derate_table;  //!< The derate table for ocv.
  Vector<std::unique_ptr<AocvLibrary>>
      _aocvs;  //!< The design aocvs of different corners.
  std::optional<StaReportSpec>
      _report_spec;  //!< The report specify for -from, -through, -to.

  std::array<std::optional<double>, TRANS_SPLIT> _max_cap;
  std::array<std::optional<double>, TRANS_SPLIT> _max_slew;
  std::optional<double> _max_fanout;

  StaGraph _graph;  //!< The graph mapped to netlist.
  std::map<Net*, std::unique_ptr<RcNet>>
      _net_to_rc_net;                         //!< The net to rc net.
  Vector<std::unique_ptr<StaClock>> _clocks;  //!< The clock domain.
  Multimap<StaVertex*, SdcSetIODelay*>
      _io_delays;  //!< The port vertex io delay constrain.
  std::map<StaClock*, std::unique_ptr<StaSeqPathGroup>>
      _clock_groups;  //!< The clock path groups.

  std::unique_ptr<StaClockGatePathGroup>
      _clock_gate_group;  //!< The clock gate path groups.

  unsigned _significant_digits =
      3;  //!< The significant digits for report, default is 3.

  std::unique_ptr<StaReportTable>
      _report_tbl_summary;  //!< The sta report table.
  std::unique_ptr<StaReportTable>
      _report_tbl_TNS;  //!< The sta report clock TNS table.
  std::vector<std::unique_ptr<StaReportTable>>
      _report_tbl_details;  //!< The sta report path detail tables.

  std::vector<std::unique_ptr<StaClockTree>>
      _clock_trees;  //!< The sta clock tree for GUI.

  std::mutex _mt;

  // Singleton sta.
  static Sta* _sta;

  DISALLOW_COPY_AND_ASSIGN(Sta);
};

}  // namespace ista
