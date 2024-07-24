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
#ifndef SRC_EVALUATION_API_EVALAPI_HPP_
#define SRC_EVALUATION_API_EVALAPI_HPP_

#include <string>
#include <vector>

#include "CongestionEval.hpp"
#include "TimingEval.hpp"
#include "WirelengthEval.hpp"
#include "ids.hpp"

namespace eval {

#define EvalInst (eval::EvalAPI::getInst())

using std::pair;
using std::string;
using std::vector;

class EvalAPI
{
 public:
  static EvalAPI& initInst();
  static EvalAPI& getInst();
  static void destroyInst();

  /****************************** Wirelength Eval: START ******************************/
  void initWLDataFromIDB();
  int64_t evalTotalWL(WIRELENGTH_TYPE wl_type);
  void plotFlowValue(const string& plot_path, const string& output_file_name, const string& step, const string& value);

  int64_t evalTotalWL(const string& wl_type);
  int64_t evalTotalWL(const string& wl_type, const vector<WLNet*>& net_list);
  int64_t evalOneNetWL(const string& wl_type, WLNet* wl_net);
  int64_t evalDriver2LoadWL(WLNet* wl_net, const string& sink_pin_name);
  double evalEGRWL();
  void reportWirelength(const string& plot_path, const string& output_file_name, const vector<WLNet*>& net_list);
  /****************************** Wirelength Eval: END *******************************/

  /****************************** Congestion Eval: START ******************************/
  void initCongDataFromIDB(const int bin_cnt_x, const int bin_cnt_y);
  void evalInstDens(INSTANCE_STATUS inst_status, bool eval_flip_flop = false);
  void evalPinDens(INSTANCE_STATUS inst_status, int level = 0);
  void evalNetDens(INSTANCE_STATUS inst_status);
  void evalLocalNetDens();
  void evalGlobalNetDens();
  void plotBinValue(const string& plot_path, const string& output_file_name, CONGESTION_TYPE cong_type);
  int32_t evalInstNum(INSTANCE_STATUS inst_status);
  int32_t evalNetNum(NET_CONNECT_TYPE net_type);
  int32_t evalPinNum(INSTANCE_STATUS inst_status = INSTANCE_STATUS::kNone);
  int32_t evalRoutingLayerNum();
  int32_t evalTrackNum(DIRECTION direction = DIRECTION::kNone);
  vector<int64_t> evalChipWidthHeightArea(CHIP_REGION_TYPE chip_region_type);
  vector<pair<string, pair<int32_t, int32_t>>> evalInstSize(INSTANCE_STATUS inst_status);
  vector<pair<string, pair<int32_t, int32_t>>> evalNetSize();
  void evalNetCong(RUDY_TYPE rudy_type, DIRECTION direction = DIRECTION::kNone);
  vector<float> evalGRCong();
  void plotTileValue(const string& plot_path, const string& output_file_name);
  float evalAreaUtils(INSTANCE_STATUS inst_status);
  int64_t evalArea(INSTANCE_STATUS inst_status);
  vector<int64_t> evalMacroPeriBias();
  int32_t evalRmTrackNum();
  int32_t evalOfTrackNum();
  int32_t evalMacroGuidance(int32_t cx, int32_t cy, int32_t width, int32_t height, const string& name);
  double evalMacroChannelUtil(float dist_ratio);
  double evalMacroChannelPinRatio(float dist_ratio);
  vector<MacroVariant> evalMacrosInfo();
  void plotMacroChannel(float dist_ratio, const std::string& filename);
  void evalMacroMargin();
  double evalMaxContinuousSpace();
  void evalIOPinAccess(const std::string& filename);
  void evalMacroDens();
  void evalMacroPinDens();
  void evalCellPinDens();
  void evalMacroChannel(float die_size_ratio = 0.5);
  void evalCellHierarchy(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroHierarchy(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroConnection(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroPinConnection(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroIOPinConnection(const std::string& plot_path, int level = 1, int forward = 1);

  vector<float> evalPinDens();
  vector<float> evalPinDens(CongGrid* grid, const vector<CongInst*>& inst_list);
  vector<float> evalInstDens();
  vector<float> evalInstDens(CongGrid* grid, const vector<CongInst*>& inst_list);
  vector<float> evalNetCong(const string& rudy_type);
  vector<float> evalNetCong(CongGrid* grid, const vector<CongNet*>& net_list, const string& rudy_type);
  // pair<vector<float>, vector<float>> evalHVNetCong();
  vector<float> getUseCapRatioList();
  vector<int> getTileGridCoordSizeCntXY();
  void plotGRCong(const string& plot_path, const string& output_file_name);
  void plotOverflow(const string& plot_path, const string& output_file_name);
  void reportCongestion(const string& plot_path, const string& output_file_name, const vector<CongNet*>& net_list, CongGrid* grid,
                        const vector<CongInst*>& inst_list);
  /****************************** Congestion Eval: END *******************************/

  /****************************** Timing Eval: START ******************************/
  void initTimingDataFromIDB();

  void initTimingEval(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, vector<const char*> lib_file_path_list,
                      const char* sdc_file_path);
  void initTimingEval(int32_t unit);
  double getEarlySlack(const string& pin_name) const;
  double getLateSlack(const string& pin_name) const;
  double getArrivalEarlyTime(const string& pin_name) const;
  double getArrivalLateTime(const string& pin_name) const;
  double getRequiredEarlyTime(const string& pin_name) const;
  double getRequiredLateTime(const string& pin_name) const;
  double reportWNS(const char* clock_name, ista::AnalysisMode mode);
  double reportTNS(const char* clock_name, ista::AnalysisMode mode);
  void updateTiming(const vector<TimingNet*>& timing_net_list);
  void updateTiming(const vector<TimingNet*>& timing_net_list, const vector<string>& name_list, const int& propagation_level);
  void destroyTimingEval();
  /****************************** Timing Eval: END *******************************/

 private:
  static EvalAPI* _eval_api_inst;
  WirelengthEval* _wirelength_eval_inst = nullptr;
  TimingEval* _timing_eval_inst = nullptr;
  CongestionEval* _congestion_eval_inst = nullptr;

  EvalAPI()
  {
    // _wirelength_eval_inst = new WirelengthEval();
    // _timing_eval_inst = new TimingEval();
    _congestion_eval_inst = new CongestionEval();
  }
  EvalAPI(const EvalAPI& other) = delete;
  EvalAPI(EvalAPI&& other) = delete;
  ~EvalAPI()
  {
    delete _wirelength_eval_inst;
    delete _timing_eval_inst;
    delete _congestion_eval_inst;
  }
  EvalAPI& operator=(const EvalAPI& other) = delete;
  EvalAPI& operator=(EvalAPI&& other) = delete;
};

}  // namespace eval

#endif  // SRC_EVALUATION_API_EVALAPI_HPP_
