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
 * @file TimingEngine.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-08-20
 */

#pragma once

#include <optional>

#include "TimingDBAdapter.hh"
#include "TimingIDBAdapter.hh"
#include "sta/Sta.hh"
#include "sta/StaIncremental.hh"

namespace ista {

class RctNode;

class TimingEngine {
 public:
  /**
   * @brief The net info of the same timing path.
   *
   */
  struct PathNet {
    StaVertex *driver;
    StaVertex *load;
    double delay;
    bool operator==(const TimingEngine::PathNet &rhs) const {
      return driver == rhs.driver && load == rhs.load && delay == rhs.delay;
    }
  };

  enum class PropType {
    kFwd,
    kBwd,
    kFwdAndBwd,
  };

  static TimingEngine *getOrCreateTimingEngine();
  static void destroyTimingEngine();

  Sta *get_ista() { return _ista; }

  Netlist *get_netlist() { return _ista->get_netlist(); }
  void writeVerilog(const char *verilog_file_name,
                    std::set<std::string> &&exclude_cell_names = {}) {
    _ista->writeVerilog(verilog_file_name, exclude_cell_names);
  }

  // Builder
  TimingEngine &set_num_threads(unsigned num_thread);

  void set_design_work_space(const char *design_work_space) {
    _ista->set_design_work_space(design_work_space);
  }
  const char *get_design_work_space() { return _ista->get_design_work_space(); }

  TimingDBAdapter *get_db_adapter() { return _db_adapter.get(); }
  auto* getIDBAdapter() { return dynamic_cast<TimingIDBAdapter*>(_db_adapter.get()); }
  void set_db_adapter(std::unique_ptr<TimingDBAdapter> db_adapter);

  TimingEngine &readLiberty(std::vector<std::string> &lib_files) {
    _ista->readLiberty(lib_files);
    return *this;
  }

  TimingEngine &readLiberty(std::vector<const char *> &lib_files) {
    std::vector<std::string> tmp;
    for (const auto *lib_file : lib_files) {
      tmp.emplace_back(lib_file);
    }

    _ista->readLiberty(tmp);
    return *this;
  }

  TimingEngine &readDesign(const char *verilog_file) {
    _ista->readVerilogWithRustParser(verilog_file);
    return *this;
  }
  TimingEngine &linkDesign(const char *top_cell_name) {
    _ista->linkDesignWithRustParser(top_cell_name);
    return *this;
  }

  TimingEngine &readDefDesign(std::string def_file, std::vector<std::string>& lef_files);
  TimingEngine &setDefDesignBuilder(void* db_builder);

  TimingEngine &readSdc(const char *sdc_file) {
    _ista->resetConstraint();
    _ista->readSdc(sdc_file);
    return *this;
  }

  TimingEngine &readSpef(const char *spef_file) {
    _ista->readSpef(spef_file);
    return *this;
  }

  TimingEngine &readAocv(std::vector<std::string> &aocv_files) {
    _ista->readAocv(aocv_files);
    return *this;
  }

  TimingEngine &readAocv(std::vector<const char *> &aocv_files) {
    std::vector<std::string> tmp;
    for (const auto *aocv_file : aocv_files) {
      tmp.emplace_back(aocv_file);
    }

    _ista->readAocv(tmp);
    return *this;
  }

  void makeClassifiedCells(std::vector<LibLibrary *> &equiv_libs) {
    return _ista->makeClassifiedCells(equiv_libs);
  }

  Vector<LibCell *> *classifyCells(LibCell *cell) {
    return _ista->classifyCells(cell);
  }

  Vector<std::unique_ptr<LibLibrary>> &getAllLib() {
    return _ista->getAllLib();
  }

  LibCell *findLibertyCell(const char *cell_name) {
    return _ista->findLibertyCell(cell_name);
  }

  LibTable *getCellLibertyTable(const char *cell_name,
                                LibTable::TableType table_type);
  LibTable *getCellLibertyTable(const char *cell_name,
                                const char *from_port_name,
                                const char *to_port_name,
                                LibTable::TableType table_type);
  LibTable *getCellLibertyTable(const char *cell_name,
                                LibArc::TimingType timing_type,
                                LibTable::TableType table_type);

  StaVertex *findVertex(const char *pin_name) {
    return _ista->findVertex(pin_name);
  }
  std::set<std::string> findStartOrEnd(const char *pin_name);
  std::map<std::string, std::string> getStartEndPairs();
  std::vector<std::tuple<std::string, std::string, double>>
  getStartEndSlackPairsOfTopNPaths(int top_n, AnalysisMode mode,
                                   TransType trans_type) {
    return _ista->getStartEndSlackPairsOfTopNPaths(top_n, mode, trans_type);
  }
  std::vector<std::tuple<std::string, std::string, double>>
  getStartEndSlackPairsOfTopNPercentPaths(double top_percentage,
                                          AnalysisMode mode,
                                          TransType trans_type) {
    return _ista->getStartEndSlackPairsOfTopNPercentPaths(top_percentage, mode,
                                                          trans_type);
  }
  std::string findClockPinName(const char *inst_name);

  void setIdealClockNetworkLatency(const char *clock_name, double latency) {
    _ista->setIdealClockNetworkLatency(clock_name, latency);
  }

  TimingEngine &resetNetlist() {
    _ista->resetNetlist();
    return *this;
  }

  TimingEngine &resetGraph() {
    _ista->resetGraph();
    return *this;
  }

  unsigned buildGraph() {
    unsigned is_ok = _ista->buildGraph();
    return is_ok;
  }

  bool isBuildGraph() {
    bool is_ok = _ista->isBuildGraph();
    return is_ok;
  }

  TimingEngine &resetGraphData() {
    _ista->resetGraphData();
    return *this;
  }

  TimingEngine &resetPathData() {
    _ista->resetPathData();
    return *this;
  }

  auto &getClockTrees() {
    auto &clock_trees = _ista->get_clock_trees();
    return clock_trees;
  }
  TimingEngine &buildClockTrees() {
    _ista->buildClockTrees();
    return *this;
  }
  // read spef
  TimingEngine &buildRCTree(const char *spef_file, DelayCalcMethod kmethod);
  void initRcTree(Net *net);
  void initRcTree();
  void resetRcTree(Net *net);
  RctNode *makeOrFindRCTreeNode(Net *net, int64_t id);
  RctNode *makeOrFindRCTreeNode(DesignObject *pin_or_port);
  RctNode* findRCTreeNode(Net *net, std::string& node_name);
  void incrCap(RctNode *node, double cap, bool is_incremental = false);
  void makeResistor(Net *net, RctNode *from_node, RctNode *to_node, double res);
  void updateRCTreeInfo(Net *net);
  void updateAllRCTree();
  void buildRcTreeAndUpdateRcTreeInfo(
      const char *net_name, std::map<std::string, double> &loadname2wl);

  std::map<std::string, double> getVirtualRCTreeAllNodeSlew(
      const char* rc_tree_name, double driver_slew, TransType trans_type);
  std::map<std::string, double> getVirtualRCTreeAllNodeDelay(
      const char* rc_tree_name);

  TimingEngine &incrUpdateTiming();

  TimingEngine &updateTiming() {
    updateAllRCTree();
    _ista->updateTiming();
    return *this;
  }

  TimingEngine &updateClockTiming() {
    _ista->updateClockTiming();
    return *this;
  }

  TimingEngine &setSignificantDigits(unsigned significant_digits) {
    _ista->set_significant_digits(significant_digits);
    return *this;
  }
  TimingEngine &reportTiming(std::set<std::string> &&exclude_cell_names = {},
                             bool is_derate = false, bool is_clock_cap = false,
                             bool is_copy = true) {
    _ista->reportTiming(std::move(exclude_cell_names), is_derate, is_clock_cap,
                        is_copy);
    return *this;
  }

  unsigned reportWirePaths(unsigned n_worst_path_per_clock) {
    _ista->set_n_worst_path_per_clock(n_worst_path_per_clock);
    return  _ista->reportWirePaths();
  }

  std::vector<StaClock *> getClockList();
  void setPropagatedClock(const char *clock_name);
  bool isPropagatedClock(const char *clock_name);

  // for any clock randomly if more than one.
  StaClock *getPropClockOfNet(Net *clock_net);
  // for all propagated clock for clock mux maybe more than once.
  std::unordered_set<StaClock *> getPropClocksOfNet(Net *clock_net);
  std::string getMasterClockOfGenerateClock(const std::string &generate_clock);
  std::string getMasterClockOfNet(Net *clock_net);
  std::vector<std::string> getMasterClocksOfNet(Net *clock_net);
  std::vector<std::string> getClockNetNameList();
  bool isClockNet(const char *net_name);
  Net *findNet(const char *net_name);

  void insertBuffer(const char *instance_name);
  void removeBuffer(const char *instance_name);
  void repowerInstance(const char *instance_name, const char *cell_name);
  void moveInstance(const char *instance_name,
                    std::optional<unsigned> update_level = std::nullopt,
                    PropType prop_type = PropType::kFwdAndBwd);

  void setNetDelay(double wl, double ucap, const char *net_name,
                   const char *load_pin_name, ModeTransPair mode_trans);

  // reporter
  double getInstDelay(const char *inst_name, const char *src_port_name,
                      const char *snk_port_name, AnalysisMode mode,
                      TransType trans_type);
  double getInstWorstArcDelay(const char *inst_name, AnalysisMode mode,
                              TransType trans_type);
  double getNetDelay(const char *net_name, const char *load_pin_name,
                     AnalysisMode mode, TransType trans_type);
  double getSlew(const char *pin_name, AnalysisMode mode, TransType trans_type);
  std::optional<double> getAT(const char *pin_name, AnalysisMode mode,
                              TransType trans_type);
  std::optional<double> getClockAT(
      const char *pin_name, AnalysisMode mode, TransType trans_type,
      std::optional<std::string> clock_name = std::nullopt);
  std::optional<double> getRT(const char *pin_name, AnalysisMode mode,
                              TransType trans_type);
  StaClock *getPropClock(const char *pin_name, AnalysisMode mode,
                         TransType trans_type);
  std::optional<double> getSlack(const char *pin_name, AnalysisMode mode,
                                 TransType trans_type);
  void getWorstSlack(AnalysisMode mode, TransType trans_type,
                     StaVertex *&worst_vertex,
                     std::optional<double> &worst_slack);
  double getWNS(const char *clock_name, AnalysisMode mode) {
    return _ista->getWNS(clock_name, mode);
  }
  double getTNS(const char *clock_name, AnalysisMode mode) {
    return _ista->getTNS(clock_name, mode);
  }
  double getLocalSkew(const char *clock_name, AnalysisMode mode,
                      TransType trans_type) {
    return _ista->getLocalSkew(clock_name, mode, trans_type);
  }
  double getGlobalSkew(AnalysisMode mode, TransType trans_type) {
    return _ista->getGlobalSkew(mode, trans_type);
  }
  std::map<StaVertex *, int> getFFMaxSkew(AnalysisMode mode,
                                          TransType trans_type) {
    return _ista->getFFMaxSkew(mode, trans_type);
  }
  std::map<StaVertex *, int> getFFTotalSkew(AnalysisMode mode,
                                            TransType trans_type) {
    return _ista->getFFTotalSkew(mode, trans_type);
  }
  std::multimap<std::string, std::string> getSkewRelatedSink(
      AnalysisMode mode, TransType trans_type) {
    return _ista->getSkewRelatedSink(mode, trans_type);
  }
  double getClockNetworkLatency(const char *clock_pin_name, AnalysisMode mode,
                                TransType trans_type);
  double getClockSkew(const char *src_clock_pin_name,
                      const char *snk_clock_pin_name, AnalysisMode mode,
                      TransType trans_type);
  double getInstPinCapacitance(const char *inst_pin_name);
  double getInstPinCapacitance(const char *inst_pin_name, AnalysisMode mode,
                               TransType trans_type);
  double getLibertyCellPinCapacitance(const char *cell_pin_name);
  std::vector<std::string> getLibertyCellInputpin(const char *cell_name);
  StaClock *getPropClock(const char *clock_pin_name);
  StaSeqPathData *getWorstSeqData(StaVertex *vertex, AnalysisMode mode,
                                  TransType trans_type) {
    return _ista->getWorstSeqData(vertex, mode, trans_type);
  }

  StaSeqPathData *getWorstSeqData(AnalysisMode mode, TransType trans_type) {
    return _ista->getWorstSeqData(std::nullopt, mode, trans_type);
  }

  double getWorstArriveTime(AnalysisMode mode = AnalysisMode::kMax) {
    double rise_AT = getWorstSeqData(mode, TransType::kRise)->getArriveTimeNs();
    double fall_AT = getWorstSeqData(mode, TransType::kFall)->getArriveTimeNs();
    double worst_AT = (rise_AT >= fall_AT ? rise_AT : fall_AT);
    return worst_AT;
  }

  std::priority_queue<StaSeqPathData *, std::vector<StaSeqPathData *>,
                      decltype(seq_data_cmp)>
  getViolatedSeqPathsBetweenTwoSinks(const char *pin1_name,
                                     const char *pin2_name, AnalysisMode mode);
  std::optional<double> getWorstSlackBetweenTwoSinks(
      const char *clock_pin1_name, const char *clock_pin2_name,
      AnalysisMode mode);
  std::map<std::pair<StaVertex *, StaVertex *>, double>
  getWorstSlackBetweenTwoSinks(AnalysisMode mode) {
    return _ista->getWorstSlackBetweenTwoSinks(mode);
  }
  std::vector<PathNet> getPathDriverVertexs(StaSeqPathData *path_data);
  int getFanoutNumOfDriverVertex(StaVertex *driver_vertex);
  std::vector<StaVertex *> getFanoutVertexs(StaVertex *driver_vertex);

  unsigned isSequentialCell(const char *instance_name);
  std::string getCellType(const char *cell_name);
  double getCellArea(const char *cell_name);
  unsigned isClock(const char *pin_name) const;
  unsigned isLoad(const char *pin_name) const;
  void validateCapacitance(const char *pin_name, AnalysisMode mode,
                           TransType trans_type, double &capacitance,
                           std::optional<double> &limit, double &slack);
  void validateFanout(const char *pin_name, AnalysisMode mode, double &fanout,
                      std::optional<double> &limit, double &slack);
  void validateSlew(const char *pin_name, AnalysisMode mode,
                    TransType trans_type, double &slew,
                    std::optional<double> &limit, double &slack);

 private:
  TimingEngine();
  ~TimingEngine();

  Sta *_ista;

  std::unique_ptr<TimingDBAdapter> _db_adapter;
  StaIncremental _incr_func;

  // Singleton timing engine.
  static TimingEngine *_timing_engine;

  FORBIDDEN_COPY(TimingEngine);
};

}  // namespace ista
