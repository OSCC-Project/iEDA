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
/**
 * @file CTSAPI.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <any>
#include <cassert>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <optional>

#include "../../../database/interaction/ids.hpp"

namespace ieda_feature {
class CTSSummary;
class ClockTiming;
}  // namespace ieda_feature

namespace icts {

#define CTSAPIInst (icts::CTSAPI::getInst())

using ieda::Log;
using ieda::Str;
using ieda::Time;
using SkewConstraintsMap = std::map<std::pair<std::string, std::string>, std::pair<double, double>>;

template <typename T>
concept StringAble = requires(const T& t) {
  { std::to_string(t) } -> std::convertible_to<std::string>;
};

class CTSAPI
{
 public:
  static CTSAPI& getInst();
  static void destroyInst();
  // open API
  void runCTS();
  void writeDB();
  void writeGDS();
  void report(const std::string& save_dir);

  // Eval Flow API
  void initEvalInfo();
  size_t getInsertCellNum() const;
  double getInsertCellArea() const;
  std::vector<PathInfo> getPathInfos() const;
  double getMaxClockNetWL() const;
  double getTotalClockNetWL() const;

  // flow API
  void resetAPI();
  void init(const std::string& config_file, const std::string& work_dir = "");
  void readData();
  void routing();
  void evaluate();
  icts::CtsConfig* get_config() { return _config; }
  icts::CtsDesign* get_design() { return _design; }
  icts::CtsDBWrapper* get_db_wrapper() { return _db_wrapper; }

  // iSTA
  void dumpVertexData(const std::vector<std::string>& vertex_names) const;
  double getClockUnitCap(const std::optional<icts::LayerPattern>& layer_pattern = std::nullopt) const;
  double getClockUnitRes(const std::optional<icts::LayerPattern>& layer_pattern = std::nullopt) const;
  double getSinkCap(icts::CtsInstance* sink) const;
  double getSinkCap(const std::string& load_pin_full_name) const;
  bool isFlipFlop(const std::string& inst_name) const;
  bool isClockNet(const std::string& net_name) const;
  void startDbSta();
  void readClockNetNames() const;
  void setPropagateClock();
  void convertDBToTimingEngine();
  void reportTiming() const;
  void refresh();
  icts::CtsPin* findDriverPin(icts::CtsNet* net);
  std::map<std::string, double> elmoreDelay(const icts::EvalNet& eval_net);
  bool cellLibExist(const std::string& cell_master, const std::string& query_field = "cell_rise", const std::string& from_port = "",
                    const std::string& to_port = "");
  std::vector<std::vector<double>> queryCellLibIndex(const std::string& cell_master, const std::string& query_field,
                                                     const std::string& from_port = "", const std::string& to_port = "");
  std::vector<double> queryCellLibValue(const std::string& cell_master, const std::string& query_field, const std::string& from_port = "",
                                        const std::string& to_port = "");
  icts::CtsCellLib* getCellLib(const std::string& cell_masterconst, const std::string& from_port = "", const std::string& to_port = "",
                               const bool& use_work_value = true);
  std::vector<icts::CtsCellLib*> getAllBufferLibs();
  icts::CtsCellLib* getRootBufferLib();
  std::vector<std::string> getMasterClocks(icts::CtsNet* net) const;
  double getClockAT(const std::string& pin_name, const std::string& belong_clock_name) const;
  std::string getCellType(const std::string& cell_master) const;
  double getCellArea(const std::string& cell_master) const;
  double getCellCap(const std::string& cell_master) const;
  double getSlewIn(const std::string& pin_name) const;
  double getCapOut(const std::string& pin_name) const;
  std::vector<double> solvePolynomialRealRoots(const std::vector<double>& coeffs);
  ieda_feature::CTSSummary outputSummary();

  // synthesis
  int32_t getDbUnit() const;
  bool isInDie(const icts::Point& point) const;
  idb::IdbInstance* makeIdbInstance(const std::string& inst_name, const std::string& cell_master);
  idb::IdbNet* makeIdbNet(const std::string& net_name);
  void linkIdbNetToSta(idb::IdbNet* idb_net);
  void disconnect(idb::IdbPin* pin);
  void connect(idb::IdbInstance* idb_inst, const std::string& pin_name, idb::IdbNet* net);
  void insertBuffer(const std::string& name);
  void resetId();
  int genId();
  void genFluteTree(const std::string& net_name, icts::Pin* driver, const std::vector<icts::Pin*>& loads);
  void genShallowLightTree(const std::string& net_name, icts::Pin* driver, const std::vector<icts::Pin*>& loads);
  icts::Inst* genBoundSkewTree(const std::string& net_name, const std::vector<icts::Pin*>& loads, const std::optional<double>& skew_bound,
                               const std::optional<icts::Point>& guide_loc, const TopoType& topo_type);
  icts::Inst* genBstSaltTree(const std::string& net_name, const std::vector<icts::Pin*>& loads, const std::optional<double>& skew_bound,
                             const std::optional<icts::Point>& guide_loc, const TopoType& topo_type);
  icts::Inst* genCBSTree(const std::string& net_name, const std::vector<icts::Pin*>& loads, const std::optional<double>& skew_bound,
                         const std::optional<icts::Point>& guide_loc, const TopoType& topo_type);
  // evaluate
  bool isTop(const std::string& net_name) const;
  void buildRCTree(const std::vector<icts::EvalNet>& eval_nets);
  void buildRCTree(const icts::EvalNet& eval_net);
  void buildPinPortsRCTree(const icts::EvalNet& eval_net);
  void resetRCTree(const std::string& net_name);
  void utilizationLog() const;
  void latencySkewLog() const;
  void slackLog() const;
  // log
  void checkFile(const std::string& dir, const std::string& file_name, const std::string& suffix = ".rpt") const;

  template <StringAble T>
  std::string stringify(const T& t)
  {
    return std::to_string(t);
  }

  std::string stringify(const std::string_view sv) { return std::string(sv); }

  template <typename... Args>
  std::string toString(const Args&... args)
  {
    return (stringify(args) + ...);
  }

  template <typename... Args>
  void saveToLog(const Args&... args)
  {
    (*_log_ofs) << toString(args...) << std::endl;
  }

  void logTime() const;

  void logLine() const;

  void logTitle(const std::string& title) const;

  void logEnd() const;

  // function
  std::vector<std::string> splitString(std::string str, const char split);

  // debug
  void writeVerilog() const;
  void toPyArray(const icts::Point& point, const std::string& label);

 private:
  static CTSAPI* _cts_api_instance;
  CTSAPI() = default;
  CTSAPI(const CTSAPI& other) = delete;
  CTSAPI(CTSAPI&& other) = delete;
  ~CTSAPI() = default;
  CTSAPI& operator=(const CTSAPI& other) = delete;
  CTSAPI& operator=(CTSAPI&& other) = delete;
  // private STA
  void readSTAFile();
  ista::RctNode* makeRCTreeNode(const icts::EvalNet& eval_net, const std::string& name);
  ista::RctNode* makePinRCTreeNode(icts::CtsPin* pin);
  ista::DesignObject* findStaPin(icts::CtsPin* pin) const;
  ista::DesignObject* findStaPin(const std::string& pin_full_name) const;
  ista::Net* findStaNet(const icts::EvalNet& eval_net) const;
  ista::Net* findStaNet(const std::string& name) const;
  double getCapacitance(const double& wire_length, const int& level) const;
  double getResistance(const double& wire_length, const int& level) const;
  ista::TimingIDBAdapter* getStaDbAdapter() const;
  void writeSkewMap() const;

  // variable
  icts::CtsConfig* _config = nullptr;
  icts::CtsDesign* _design = nullptr;
  icts::CtsDBWrapper* _db_wrapper = nullptr;
  icts::CtsReportTable* _report = nullptr;
  std::ofstream* _log_ofs = nullptr;
  icts::CtsLibs* _libs = nullptr;
  icts::Evaluator* _evaluator = nullptr;
  icts::ModelFactory* _model_factory = nullptr;
  ista::TimingEngine* _timing_engine = nullptr;
};

}  // namespace icts
