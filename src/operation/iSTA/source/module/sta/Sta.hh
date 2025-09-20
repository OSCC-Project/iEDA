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
#include <shared_mutex>
#include <string>
#include <utility>

#include "FlatMap.hh"
#include "StaClock.hh"
#include "StaClockTree.hh"
#include "StaGraph.hh"
#include "StaPathData.hh"
#include "StaReport.hh"
#include "Type.hh"
#include "aocv/AocvParser.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "json/json.hpp"
#include "liberty/Lib.hh"
#include "liberty/LibClassifyCell.hh"
#include "netlist/Netlist.hh"
#include "sdc/SdcSetIODelay.hh"
#include "verilog/VerilogParserRustC.hh"

#if CUDA_PROPAGATION
#include "propagation-cuda/fwd_propagation.cuh"
#include "propagation-cuda/propagation.cuh"
#endif

namespace ista {

class SdcConstrain;

constexpr int g_global_derate_num = 8;

// minHeap of the StaSeqPathData.
const std::function<bool(StaSeqPathData*, StaSeqPathData*)> seq_data_cmp =
    [](StaSeqPathData* left, StaSeqPathData* right) -> bool {
  unsigned left_slack = left->getSlack();
  unsigned right_slack = right->getSlack();
  return left_slack > right_slack;
};

// clock cmp for staclock.
const std::function<unsigned(StaClock*, StaClock*)> sta_clock_cmp =
    [](StaClock* left, StaClock* right) -> unsigned {
  return Str::caseCmp(left->get_clock_name(), right->get_clock_name()) < 0;
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

  void addLinkCells(std::set<std::string> link_cells) {
    _link_cells.insert(link_cells.begin(), link_cells.end());
  }
  auto& get_link_cells() { return _link_cells; }

  auto get_propagation_method() { return _propagation_method; }

  SdcConstrain* getConstrain();

  unsigned readDesignWithRustParser(const char* verilog_file);
  unsigned readLiberty(const char* lib_file);
  unsigned readLiberty(std::vector<std::string>& lib_files);
  unsigned readSdc(const char* sdc_file);
  unsigned readSpef(const char* spef_file);
  unsigned readAocv(const char* aocv_file);
  unsigned readAocv(std::vector<std::string>& aocv_files);

  void set_top_module_name(const char* top_module_name) {
    _top_module_name = top_module_name;
  }
  auto& get_top_module_name() { return _top_module_name; }

  unsigned readVerilogWithRustParser(const char* verilog_file);
  void collectLinkedCell();
  void linkDesignWithRustParser(const char* top_cell_name);
  void set_design_name(const char* design_name) {
    _netlist.set_name(design_name);
  }
  std::string& get_design_name() { return _netlist.getObjName(); }

  auto& get_constrains() { return _constrains; }
  void resetConstraint();

  Netlist* get_netlist() { return &_netlist; }
  void resetNetlist() { _netlist.reset(); }

  void addLibReaders(RustLibertyReader lib_reader) {
    std::unique_lock<std::mutex> lk(_mt);
    _lib_readers.emplace_back(std::move(lib_reader));
  }
  auto& get_lib_readers() { return _lib_readers; }

  void addLib(std::unique_ptr<LibLibrary> lib) {
    std::unique_lock<std::mutex> lk(_mt);
    _libs.emplace_back(std::move(lib));
  }

  LibLibrary* getOneLib() {
    return _libs.empty() ? nullptr : _libs.back().get();
  }

  std::set<LibLibrary*> getUsedLibs();

  Vector<std::unique_ptr<LibLibrary>>& getAllLib() { return _libs; }

  unsigned linkLibertys();
  void resetRcNet(Net* the_net) {
    std::unique_lock<std::shared_mutex> lock(_rw_mutex);
    if (_net_to_rc_net.contains(the_net)) {
      _net_to_rc_net.erase(the_net);
    }
  }

  void addRcNet(Net* the_net, std::unique_ptr<RcNet>&& rc_net) {
    std::unique_lock<std::shared_mutex> lock(_rw_mutex);
    _net_to_rc_net[the_net] = std::move(rc_net);
  }

  void removeRcNet(Net* the_net) { _net_to_rc_net.erase(the_net); }
  RcNet* getRcNet(Net* the_net) {
    std::shared_lock<std::shared_mutex> lock(_rw_mutex);
    RcNet* rc_net = _net_to_rc_net.contains(the_net)
                        ? _net_to_rc_net[the_net].get()
                        : nullptr;

    return rc_net;
  }
  std::vector<RcNet*> getAllRcNet() {
    std::vector<RcNet*> rc_nets;
    for (auto& [net, rc_net] : _net_to_rc_net) {
      if (!rc_net->rct() || !rc_net->rct()->get_root()) {
        continue;
      }
      rc_nets.push_back(rc_net.get());
    }
    return rc_nets;
  }

  void resetAllRcNet() { _net_to_rc_net.clear(); }
  LibCell* findLibertyCell(const char* cell_name);
  std::optional<AocvObjectSpecSet*> findDataAocvObjectSpecSet(
      const char* object_name);
  std::optional<AocvObjectSpecSet*> findClockAocvObjectSpecSet(
      const char* object_name);
  void makeClassifiedCells(std::vector<LibLibrary*>& equiv_libs);

  Vector<LibCell*>* classifyCells(LibCell* cell);

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

  std::map<StaClock*, unsigned> getClockToIndex() {
    std::map<StaClock*, unsigned> clock_to_index;
    for (size_t i = 0; i < _clocks.size(); ++i) {
      clock_to_index[_clocks[i].get()] = i;
    }
    return clock_to_index;
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

#if CUDA_PROPAGATION
  unsigned buildLibArcsGPU();
  void set_gpu_lib_data(Lib_Data_GPU&& lib_data_gpu) {
    _gpu_lib_data = std::move(lib_data_gpu);
  }
  auto& get_gpu_lib_data() { return _gpu_lib_data; }

  void set_lib_gpu_tables(std::vector<Lib_Table_GPU> lib_gpu_tables) {
    _lib_gpu_tables = std::move(lib_gpu_tables);
  }
  auto& get_lib_gpu_tables() { return _lib_gpu_tables; }

  void set_lib_gpu_table_ptr(std::vector<Lib_Table_GPU*> lib_gpu_table_ptrs) {
    _lib_gpu_table_ptrs = std::move(lib_gpu_table_ptrs);
  }
  auto& get_lib_gpu_table_ptrs() { return _lib_gpu_table_ptrs; }
  void set_lib_gpu_arcs(std::vector<Lib_Arc_GPU>&& lib_gpu_arcs) {
    _lib_gpu_arcs = std::move(lib_gpu_arcs);
  }
  auto& get_lib_gpu_arcs() { return _lib_gpu_arcs; }

  void set_gpu_graph(GPU_Graph&& the_gpu_graph) {
    _gpu_graph = std::move(the_gpu_graph);
  }
  auto& get_gpu_graph() { return _gpu_graph; }

  void set_arc_to_index(std::map<StaArc*, unsigned>&& arc_to_index) {
    _arc_to_index = std::move(arc_to_index);
  }
  auto& get_arc_to_index() { return _arc_to_index; }

  void set_index_to_at(std::map<unsigned, StaPathDelayData*>&& index_to_at) {
    _index_to_at = std::move(index_to_at);
  }
  auto& get_index_to_at() { return _index_to_at; }

  void set_at_to_index(std::map<StaPathDelayData*, unsigned>&& at_to_index) {
    _at_to_index = std::move(at_to_index);
  }
  auto& get_at_to_index() { return _at_to_index; }

  void set_gpu_vertices(std::vector<GPU_Vertex>&& gpu_vertices) {
    _gpu_vertices = std::move(gpu_vertices);
  }
  auto& get_gpu_vertices() { return _gpu_vertices; }

  void set_gpu_arcs(std::vector<GPU_Arc>&& gpu_arcs) {
    _gpu_arcs = std::move(gpu_arcs);
  }
  auto& get_gpu_arcs() { return _gpu_arcs; }

  void set_flatten_data(GPU_Flatten_Data&& flatten_data) {
    _flatten_data = std::move(flatten_data);
  }
  auto& get_flatten_data() { return _flatten_data; }

  void printFlattenData();

#endif

  StaVertex* findVertex(const char* pin_name);
  StaVertex* findVertex(DesignObject* obj) {
    auto the_vertex = _graph.findVertex(obj);
    return the_vertex ? *the_vertex : nullptr;
  }
  StaVertex* getVertex(unsigned index) {
    auto& the_vertexes = _graph.get_vertexes();
    auto vertex_size = the_vertexes.size();
    if (index < vertex_size) {
      return the_vertexes[index].get();
    } else {
      auto assistants = _graph.getAssistants();
      return assistants[index - vertex_size];
    }
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

  unsigned reportPath(const char* rpt_file_name, bool is_derate = true,
                      bool only_wire_path = false);
  unsigned reportTrans(const char* rpt_file_name);
  unsigned reportCap(const char* rpt_file_name, bool is_clock_cap);
  unsigned reportFanout(const char* rpt_file_name);
  unsigned reportSkew(const char* rpt_file_name, AnalysisMode analysis_mode);
  unsigned reportNet(const char* rpt_file_name, Net* net);
  unsigned reportNet();
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

  std::vector<StaSeqPathData*> getSeqData(StaVertex* vertex,
                                          StaData* delay_data);
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
  std::vector<StaSeqPathData*> getWorstSeqData(std::optional<StaVertex*> vertex,
                                  AnalysisMode mode, TransType trans_type,
                                  unsigned top_n_path = 1);
  StaSeqPathData* getWorstSeqData(AnalysisMode mode, TransType trans_type);
  std::vector<StaSeqPathData*> getTopNWorstSeqPaths(AnalysisMode mode,
                                                    unsigned top_n_path);

  std::vector<std::tuple<std::string, std::string, double>>
  getStartEndSlackPairsOfTopNPaths(int top_n, AnalysisMode mode,
                                   TransType trans_type);
  std::vector<std::tuple<std::string, std::string, double>>
  getStartEndSlackPairsOfTopNPercentPaths(double top_percentage,
                                          AnalysisMode mode,
                                          TransType trans_type);

  std::priority_queue<StaSeqPathData*, std::vector<StaSeqPathData*>,
                      decltype(seq_data_cmp)>
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
#if CUDA_PROPAGATION
  unsigned resetGPUData();
#endif
  unsigned updateTiming();
  unsigned updateClockTiming();
  std::set<std::string> findStartOrEnd(StaVertex* the_vertex, bool is_find_end);
  unsigned reportTiming(std::set<std::string>&& exclude_cell_names = {},
                        bool is_derate = false, bool is_clock_cap = false,
                        bool is_copy = true);

  std::vector<StaPathWireTimingData> reportTimingData(
      unsigned n_worst_path_per_clock);
  unsigned reportUsedLibs();
  unsigned reportWirePaths();

  void dumpVertexData(std::vector<std::string> vertex_names);
  void dumpNetlistData();

  void dumpGraphData(const char* graph_file);

  void buildClockTrees();

  // const char* getUnit(const char* unit_name);
  // void setUnit(const char* unit_name, char* unit_value);
  // double convertToStaUnit(const char* src_type, const double src_value);

  TimeUnit getTimeUnit() const { return _time_unit; };
  void setTimeUnit(TimeUnit new_time_unit) { _time_unit = new_time_unit; };
  double convertTimeUnit(const double src_value);

  CapacitiveUnit getCapUnit() const { return _cap_unit; };
  void setCapUnit(CapacitiveUnit new_cap_unit) { _cap_unit = new_cap_unit; };
  double convertCapUnit(const double src_value);

  std::optional<double> getInstWorstSlack(AnalysisMode analysis_mode,
                                          Instance* the_inst);
  std::optional<double> getInstTotalNegativeSlack(AnalysisMode analysis_mode,
                                                  Instance* the_inst);
  std::optional<double> getInstTransition(AnalysisMode analysis_mode,
                                          Instance* the_inst);

  std::map<Instance::Coordinate, double> displayTimingMap(
      AnalysisMode analysis_mode);
  std::map<Instance::Coordinate, double> displayTimingTNSMap(
      AnalysisMode analysis_mode);

  std::map<Instance::Coordinate, double> displayTransitionMap(
      AnalysisMode analysis_mode);

  void enableJsonReport() { _is_json_report_enabled = true; }

  bool isJsonReportEnabled() const { return _is_json_report_enabled; }

  auto& getSummaryJsonReport() { return _summary_json_report; }
  auto& getSlackJsonReport() { return _slack_json_report; }
  auto& getDetailJsonReport() { return _detail_json_report; }

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
  RustVerilogReader _rust_verilog_reader;
  void* _rust_verilog_file_ptr;
  std::string _top_module_name;
  std::vector<RustVerilogModule*>
      _rust_verilog_modules;  //!< The current design parsed from verilog file
                              //!< of rust version.
  RustVerilogModule* _rust_top_module =
      nullptr;       //!< The design top module of rust version.
  Netlist _netlist;  //!< The current top netlist for sta analysis.

  std::vector<RustLibertyReader>
      _lib_readers;  //!< The design lib parsed files.
  Vector<std::unique_ptr<LibLibrary>>
      _libs;  //!< The design libs of different corners.

  std::set<std::string>
      _link_cells;  //!< The linked cell names for liberty load.
  std::unique_ptr<LibClassifyCell>
      _classified_cells;  //!< The function equivalently liberty cell.

  AnalysisMode _analysis_mode;  //!< The analysis max/min mode.
  PropagationMethod _propagation_method =
      PropagationMethod::kBFS;  //!< The propagation method used by DFS or BFS.

  StaDreateTable _derate_table;  //!< The derate table for ocv.
  Vector<std::unique_ptr<AocvLibrary>>
      _aocvs;  //!< The design aocvs of different corners.
  std::optional<StaReportSpec>
      _report_spec;  //!< The report specify for -from, -through, -to.

  std::array<std::optional<double>, TRANS_SPLIT> _max_cap;
  std::array<std::optional<double>, TRANS_SPLIT> _max_slew;
  std::optional<double> _max_fanout;

  StaGraph _graph;  //!< The graph mapped to netlist.

  unsigned _significant_digits =
      3;  //!< The significant digits for report, default is 3.

  TimeUnit _time_unit = TimeUnit::kNS;
  CapacitiveUnit _cap_unit = CapacitiveUnit::kPF;

  std::map<Net*, std::unique_ptr<RcNet>>
      _net_to_rc_net;                         //!< The net to rc net.
  Vector<std::unique_ptr<StaClock>> _clocks;  //!< The clock domain.
  Multimap<StaVertex*, SdcSetIODelay*>
      _io_delays;  //!< The port vertex io delay constrain.
  std::map<StaClock*, std::unique_ptr<StaSeqPathGroup>, decltype(sta_clock_cmp)>
      _clock_groups;  //!< The clock path groups.

  std::unique_ptr<StaClockGatePathGroup>
      _clock_gate_group;  //!< The clock gate path groups.

  std::unique_ptr<StaReportTable>
      _report_tbl_summary;  //!< The sta report table.
  std::unique_ptr<StaReportTable>
      _report_tbl_TNS;  //!< The sta report clock TNS table.
  std::vector<std::unique_ptr<StaReportTable>>
      _report_tbl_details;  //!< The sta report path detail tables.

  std::vector<std::unique_ptr<StaClockTree>>
      _clock_trees;  //!< The sta clock tree for GUI.

  std::mutex _mt;
  std::shared_mutex _rw_mutex;  //!< For rc net.
  // Singleton sta.
  static Sta* _sta;

  using json = nlohmann::ordered_json;

  bool _is_json_report_enabled = false;        //!< The json report enable flag.
  json _summary_json_report = json::array();  //!< The json data
  json _slack_json_report = json::array();    //!< The json data
  json _detail_json_report =
      json::array();  //!< The json data for detailed report.

#if CUDA_PROPAGATION
  std::vector<GPU_Vertex> _gpu_vertices;  //!< gpu flatten vertex, arc data.
  std::vector<GPU_Arc> _gpu_arcs;
  GPU_Flatten_Data _flatten_data;
  GPU_Graph _gpu_graph;  //!< The gpu graph mapped to sta graph.
  std::vector<Lib_Arc_GPU> _lib_gpu_arcs;           //!< The gpu lib arc data.
  Lib_Data_GPU _gpu_lib_data;                       //!< The gpu lib arc data.
  std::vector<Lib_Table_GPU> _lib_gpu_tables;       //!< The gpu lib table data.
  std::vector<Lib_Table_GPU*> _lib_gpu_table_ptrs;  //!< The gpu lib table data.
  std::map<StaArc*, unsigned> _arc_to_index;  //!< The arc map to gpu index.
  std::map<StaPathDelayData*, unsigned>
      _at_to_index;  //!< The at map to gpu index.
  std::map<unsigned, StaPathDelayData*>
      _index_to_at;  //!< The gpu index to at map.
#endif

  FORBIDDEN_COPY(Sta);
};

}  // namespace ista
