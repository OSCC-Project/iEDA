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
/*
 * @Author: S.J Chen
 * @Date: 2022-10-24 21:17:18
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-11 14:20:43
 * @FilePath: /irefactor/src/operation/iPL/api/PLAPI.hh
 * @Description: Interface of iPL.
 */

#ifndef IPL_API_H
#define IPL_API_H

#include "external_api/ExternalAPI.hh"
#include "report/PLReporter.hh"

namespace ieda_feature {
class PlaceSummary;
}  // namespace ieda_feature

namespace ipl {

#define iPLAPIInst ipl::PLAPI::getInst()

class PLAPI
{
 public:
  static PLAPI& getInst();
  static void destoryInst();

  void initAPI(std::string pl_json_path, idb::IdbBuilder* idb_builder);
  void runFlow();
  void runAiFlow(const std::string& onnx_path, const std::string& normalization_path);
  void runIncrementalFlow();
  void insertLayoutFiller();

  void runGP();
  void runMP();
  void runNetworkFlowSpread();

  bool runLG();
  bool runIncrLG();
  bool runIncrLG(std::vector<std::string> inst_name_list);
  void runPostGP();
  void runDP();
#ifdef ENABLE_AI
  void runAIDP(const std::string& onnx_path, const std::string& normalization_path);
#endif
  void runBufferInsertion();
  void writeBackSourceDataBase();

  std::string obtainTargetDir();

  void updatePlacerDB();
  void updatePlacerDB(std::vector<std::string> inst_list);

  std::vector<Rectangle<int32_t>> obtainAvailableWhiteSpaceList(std::pair<int32_t, int32_t> row_range,
                                                                std::pair<int32_t, int32_t> site_range);
  bool checkLegality();

  PLReporter* get_reporter() { return _reporter; }

  void reportPLInfo();
  void reportTopoInfo();
  void reportWLInfo(std::ofstream& feed);
  void reportSTWLInfo(std::ofstream& feed);
  void reportHPWLInfo(std::ofstream& feed);
  void reportLongNetInfo(std::ofstream& feed);
  void reportViolationInfo(std::ofstream& feed);
  void reportBinDensity(std::ofstream& feed);
  int32_t reportOverlapInfo(std::ofstream& feed);
  void reportLayoutWhiteInfo();
  void reportTimingInfo(std::ofstream& feed);
  void reportCongestionInfo(std::ofstream& feed);
  void reportPLBaseInfo(std::ofstream& feed);

  void notifyPLWLInfo(int stage);  // for indicator record: 0-GP, 1-LG, 2-DP
  void notifyPLTimingInfo(int stage);
  void notifySTAUpdateTimingRuntime();
  void notifyPLCongestionInfo(int stage);
  void notifyPLOriginInfo();

  bool isSTAStarted();
  bool isPlacerDBStarted();
  bool isAbucasLGStarted();

  // The following interfaces are only for iPL internal calls !
  // The following interfaces are only for iPL internal calls !
  // The following interfaces are only for iPL internal calls !

  void createPLDirectory();
  void printHPWLInfo();
  void printTimingInfo();
  void saveNetPinInfoForDebug(std::string path);
  void savePinListInfoForDebug(std::string path);
  void plotConnectionForDebug(std::vector<std::string> net_name_list, std::string path);
  void plotModuleListForDebug(std::vector<std::string> module_prefix_list, std::string path);
  void plotModuleStateForDebug(std::vector<std::string> special_inst_list, std::string path);

  void modifySTAOutputDir(std::string path);
  void initSTA(std::string path, bool init_log);
  void updateSTATiming();
  std::vector<std::string> obtainClockNameList();
  bool isClockNet(std::string net_name);
  bool isSequentialCell(std::string inst_name);
  bool isBufferCell(std::string cell_name);
  void updateSequentialProperty();

  bool insertSignalBuffer(std::pair<std::string, std::string> source_sink_net, std::vector<std::string> sink_pin_list,
                          std::pair<std::string, std::string> master_inst_buffer, std::pair<int, int> buffer_center_loc);

  void enableJsonOutput() { _enable_json_output = true; }
  bool isJsonOutputEnabled() { return _enable_json_output; }

  /*****************************Timing-driven Placement: START*****************************/
  double obtainPinEarlySlack(std::string pin_name);
  double obtainPinLateSlack(std::string pin_name);
  double obtainPinEarlyArrivalTime(std::string pin_name);
  double obtainPinLateArrivalTime(std::string pin_name);
  double obtainPinEarlyRequiredTime(std::string pin_name);
  double obtainPinLateRequiredTime(std::string pin_name);
  double obtainWNS(const char* clock_name, ista::AnalysisMode mode);
  double obtainTNS(const char* clock_name, ista::AnalysisMode mode);
  double obtainEarlyWNS(const char* clock_name);
  double obtainEarlyTNS(const char* clock_name);
  double obtainLateWNS(const char* clock_name);
  double obtainLateTNS(const char* clock_name);
  void updateTiming(TopologyManager* topo_manager);
  void updatePartOfTiming(TopologyManager* topo_manager,
                          std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>>& net_id_to_points_map);
  void updateTimingInstMovement(TopologyManager* topo_manager,
                                std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map,
                                std::vector<std::string> moved_inst_list);
  float obtainPinCap(std::string inst_pin_name);
  float obtainAvgWireResUnitLengthUm();
  float obtainAvgWireCapUnitLengthUm();
  float obtainInstOutPinRes(std::string cell_name, std::string port_name);
  ieval::TimingNet* generateTimingNet(NetWork* network,
                                      const std::vector<std::pair<ipl::Point<int32_t>, ipl::Point<int32_t>>>& point_pair_list);
  void destroyTimingEval();

  /*****************************Timing-driven Placement: END*****************************/

  ieda_feature::PlaceSummary outputSummary(std::string step);

 private:
  static PLAPI* _s_ipl_api_instance;
  ExternalAPI* _external_api;
  PLReporter* _reporter;

  bool _enable_json_output = false;

  PLAPI() = default;
  PLAPI(const PLAPI&) = delete;
  PLAPI(PLAPI&&) = delete;
  ~PLAPI();
  PLAPI& operator=(const PLAPI&) = delete;
  PLAPI& operator=(PLAPI&&) = delete;
};

}  // namespace ipl

#endif  // IPL_API_H
