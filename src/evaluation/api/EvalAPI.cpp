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

#include "EvalAPI.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include "RTInterface.hpp"
#include "idm.h"
#include "manager.hpp"

namespace eval {

EvalAPI& EvalAPI::initInst()
{
  if (_eval_api_inst == nullptr) {
    _eval_api_inst = new EvalAPI();
  }
  return *_eval_api_inst;
}

EvalAPI& EvalAPI::getInst()
{
  if (_eval_api_inst == nullptr) {
    std::cout << "The instance not initialized!";
  }
  return *_eval_api_inst;
}

void EvalAPI::destroyInst()
{
  if (_eval_api_inst != nullptr) {
    delete _eval_api_inst;
    _eval_api_inst = nullptr;
  }
}

EvalAPI* EvalAPI::_eval_api_inst = nullptr;

/******************************Wirelength Eval: START******************************/
void EvalAPI::initWLDataFromIDB()
{
  _wirelength_eval_inst->initWLNetList();
}

int64_t EvalAPI::evalTotalWL(const string& wl_type)
{
  return _wirelength_eval_inst->evalTotalWL(wl_type);
}

int64_t EvalAPI::evalTotalWL(WIRELENGTH_TYPE wl_type)
{
  if (wl_type == WIRELENGTH_TYPE::kEGR) {
    return evalEGRWL();
  }
  return _wirelength_eval_inst->evalTotalWL(wl_type);
}

void EvalAPI::plotFlowValue(const string& plot_path, const string& output_file_name, const string& step, const string& value)
{
  string csv_file_path = plot_path + "/" + output_file_name + ".csv";

  bool is_file_exists = std::ifstream(csv_file_path).good();
  if (!is_file_exists) {
    std::ofstream csv_file(csv_file_path);
    if (csv_file.is_open()) {
      csv_file << "Step,Value\n";
      csv_file.close();
    } else {
      std::cout << "Unable to open csv file: " << csv_file_path << std::endl;
      return;
    }
  }

  std::ofstream csv_file(csv_file_path, std::ofstream::app);
  if (csv_file.is_open()) {
    csv_file << step << "," << value << "\n";
    csv_file.close();
  } else {
    std::cout << "Unable to open csv file: " << csv_file_path << std::endl;
  }
}

int64_t EvalAPI::evalTotalWL(const string& wl_type, const vector<WLNet*>& net_list)
{
  WirelengthEval wirelength_eval;
  wirelength_eval.checkWLType(wl_type);
  wirelength_eval.set_net_list(net_list);
  return wirelength_eval.evalTotalWL(wl_type);
}

int64_t EvalAPI::evalOneNetWL(const string& wl_type, WLNet* wl_net)
{
  WirelengthEval wirelength_eval;
  wirelength_eval.checkWLType(wl_type);
  return wirelength_eval.evalOneNetWL(wl_type, wl_net);
}

int64_t EvalAPI::evalDriver2LoadWL(WLNet* wl_net, const string& sink_pin_name)
{
  WirelengthEval wirelength_eval;
  return wirelength_eval.evalDriver2LoadWL(wl_net, sink_pin_name);
}

double EvalAPI::evalEGRWL()
{
  return 0;

  // call router to get eGR wirelength info
  // irt::RTI& rt_api = irt::RTI::getInst();
  // std::map<std::string, std::any> config_map;
  // std::vector<double> wl_via_pair = rt_api.getWireLengthAndViaNum(config_map);
  // rt_api.destroyInst();

  // return wl_via_pair[0];
}

void EvalAPI::reportWirelength(const string& plot_path, const string& output_file_name, const vector<WLNet*>& net_list)
{
  WirelengthEval wirelength_eval;
  wirelength_eval.set_net_list(net_list);
  wirelength_eval.reportWirelength(plot_path, output_file_name);
}
/******************************Wirelength Eval: END******************************/

/******************************Congestion Eval: START******************************/

void EvalAPI::initCongDataFromIDB(const int bin_cnt_x, const int bin_cnt_y)
{
  // initialize cong_grid
  _congestion_eval_inst->initCongGrid(bin_cnt_x, bin_cnt_y);
  // transform idb_inst to cong_inst
  _congestion_eval_inst->initCongInst();
  // tansform idb_net to cong_net
  _congestion_eval_inst->initCongNetList();
  // map cong_inst to each cong_bin
  _congestion_eval_inst->mapInst2Bin();
  // map cong_net to each cong_bin
  _congestion_eval_inst->mapNetCoord2Grid();
}

void EvalAPI::evalInstDens(INSTANCE_STATUS inst_status, bool eval_flip_flop)
{
  _congestion_eval_inst->evalInstDens(inst_status, eval_flip_flop);
}

void EvalAPI::evalMacroDens()
{
  _congestion_eval_inst->evalMacroDens();
}

void EvalAPI::evalMacroPinDens()
{
  _congestion_eval_inst->evalMacroPinDens();
}

void EvalAPI::evalCellPinDens()
{
  _congestion_eval_inst->evalCellPinDens();
}

void EvalAPI::evalMacroChannel(float die_size_ratio)
{
  _congestion_eval_inst->evalMacroChannel(die_size_ratio);
}

void EvalAPI::evalCellHierarchy(const std::string& plot_path, int level, int forward)
{
  _congestion_eval_inst->evalCellHierarchy(plot_path, level, forward);
}

void EvalAPI::evalMacroHierarchy(const std::string& plot_path, int level, int forward)
{
  _congestion_eval_inst->evalMacroHierarchy(plot_path, level, forward);
}

void EvalAPI::evalMacroConnection(const std::string& plot_path, int level, int forward)
{
  _congestion_eval_inst->evalMacroConnection(plot_path, level, forward);
}

void EvalAPI::evalMacroPinConnection(const std::string& plot_path, int level, int forward)
{
  _congestion_eval_inst->evalMacroPinConnection(plot_path, level, forward);
}

void EvalAPI::evalMacroIOPinConnection(const std::string& plot_path, int level, int forward)
{
  _congestion_eval_inst->evalMacroIOPinConnection(plot_path, level, forward);
}

void EvalAPI::evalPinDens(INSTANCE_STATUS inst_status, int level)
{
  _congestion_eval_inst->evalPinDens(inst_status, level);
}

void EvalAPI::evalNetDens(INSTANCE_STATUS inst_status)
{
  _congestion_eval_inst->evalNetDens(inst_status);
}

void EvalAPI::evalLocalNetDens()
{
  _congestion_eval_inst->evalLocalNetDens();
}

void EvalAPI::evalGlobalNetDens()
{
  _congestion_eval_inst->evalGlobalNetDens();
}

void EvalAPI::plotBinValue(const string& plot_path, const string& output_file_name, CONGESTION_TYPE cong_type)
{
  _congestion_eval_inst->plotBinValue(plot_path, output_file_name, cong_type);
}

int32_t EvalAPI::evalInstNum(INSTANCE_STATUS inst_status)
{
  return _congestion_eval_inst->evalInstNum(inst_status);
}

int32_t EvalAPI::evalNetNum(NET_CONNECT_TYPE net_type)
{
  return _congestion_eval_inst->evalNetNum(net_type);
}

int32_t EvalAPI::evalPinNum(INSTANCE_STATUS inst_status)
{
  return _congestion_eval_inst->evalPinTotalNum(inst_status);
}

int32_t EvalAPI::evalRoutingLayerNum()
{
  return _congestion_eval_inst->evalRoutingLayerNum();
}

int32_t EvalAPI::evalTrackNum(DIRECTION direction)
{
  return _congestion_eval_inst->evalTrackNum(direction);
}

vector<int64_t> EvalAPI::evalChipWidthHeightArea(CHIP_REGION_TYPE chip_region_type)
{
  return _congestion_eval_inst->evalChipWidthHeightArea(chip_region_type);
}

vector<pair<string, pair<int32_t, int32_t>>> EvalAPI::evalInstSize(INSTANCE_STATUS inst_status)
{
  return _congestion_eval_inst->evalInstSize(inst_status);
}

vector<pair<string, pair<int32_t, int32_t>>> EvalAPI::evalNetSize()
{
  return _congestion_eval_inst->evalNetSize();
}

void EvalAPI::evalNetCong(RUDY_TYPE rudy_type, DIRECTION direction)
{
  _congestion_eval_inst->evalNetCong(rudy_type, direction);
}

void EvalAPI::plotTileValue(const string& plot_path, const string& output_file_name)
{
  _congestion_eval_inst->plotTileValue(plot_path, output_file_name);
}

float EvalAPI::evalAreaUtils(INSTANCE_STATUS inst_status)
{
  return _congestion_eval_inst->evalAreaUtils(inst_status);
}

int64_t EvalAPI::evalArea(INSTANCE_STATUS inst_status)
{
  return _congestion_eval_inst->evalArea(inst_status);
}

// reference: “RTL-MP: Toward Practical, Human-Quality Chip Planning and Macro Placement”
vector<int64_t> EvalAPI::evalMacroPeriBias()
{
  return _congestion_eval_inst->evalMacroPeriBias();
}

int32_t EvalAPI::evalRmTrackNum()
{
  return _congestion_eval_inst->evalRmTrackNum();
}

int32_t EvalAPI::evalOfTrackNum()
{
  return _congestion_eval_inst->evalOfTrackNum();
}

int32_t EvalAPI::evalMacroGuidance(int32_t cx, int32_t cy, int32_t width, int32_t height, const string& name)
{
  return _congestion_eval_inst->evalMacroGuidance(cx, cy, width, height, name);
}

double EvalAPI::evalMacroChannelUtil(float dist_ratio)
{
  return _congestion_eval_inst->evalMacroChannelUtil(dist_ratio);
}

double EvalAPI::evalMacroChannelPinRatio(float dist_ratio)
{
  return _congestion_eval_inst->evalMacroChannelPinRatio(dist_ratio);
}

vector<MacroVariant> EvalAPI::evalMacrosInfo()
{
  return _congestion_eval_inst->evalMacrosInfo();
}

void EvalAPI::plotMacroChannel(float dist_ratio, const std::string& filename)
{
  _congestion_eval_inst->plotMacroChannel(dist_ratio, filename);
}

void EvalAPI::evalMacroMargin()
{
  _congestion_eval_inst->evalMacroMargin();
}

double EvalAPI::evalMaxContinuousSpace()
{
  return _congestion_eval_inst->evalMaxContinuousSpace();
}

void EvalAPI::evalIOPinAccess(const std::string& filename)
{
  return _congestion_eval_inst->evalIOPinAccess(filename);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
vector<float> EvalAPI::evalPinDens()
{
  return _congestion_eval_inst->evalPinDens();
}

vector<float> EvalAPI::evalPinDens(CongGrid* grid, const vector<CongInst*>& inst_list)
{
  _congestion_eval_inst->set_cong_grid(grid);
  _congestion_eval_inst->set_cong_inst_list(inst_list);
  _congestion_eval_inst->mapInst2Bin();
  return _congestion_eval_inst->evalPinDens();
}

vector<float> EvalAPI::evalInstDens()
{
  _congestion_eval_inst->mapInst2Bin();
  return _congestion_eval_inst->getInstDens();
}

vector<float> EvalAPI::evalInstDens(CongGrid* grid, const vector<CongInst*>& inst_list)
{
  _congestion_eval_inst->set_cong_grid(grid);
  _congestion_eval_inst->set_cong_inst_list(inst_list);
  _congestion_eval_inst->mapInst2Bin();
  return _congestion_eval_inst->getInstDens();
}

vector<float> EvalAPI::evalNetCong(const string& rudy_type)
{
  _congestion_eval_inst->checkRUDYType(rudy_type);
  _congestion_eval_inst->mapNetCoord2Grid();
  return _congestion_eval_inst->getNetCong(rudy_type);
}

vector<float> EvalAPI::evalNetCong(CongGrid* grid, const vector<CongNet*>& net_list, const string& rudy_type)
{
  CongestionEval congestion_eval;
  congestion_eval.checkRUDYType(rudy_type);
  congestion_eval.set_cong_grid(grid);
  congestion_eval.set_cong_net_list(net_list);
  congestion_eval.mapNetCoord2Grid();
  return congestion_eval.getNetCong(rudy_type);
}

vector<float> EvalAPI::evalGRCong()
{
  return {};
  // // call router to get tilegrid info
  // irt::RTI& rt_api = irt::RTI::getInst();
  // std::map<std::string, std::any> config_map;
  // double wirelength = 0.0;
  // TileGrid* tile_grid = rt_api.getCongestionMap(config_map, wirelength);
  // rt_api.destroyInst();

  // _congestion_eval_inst->set_tile_grid(tile_grid);

  // vector<float> result;
  // result.reserve(4);
  // result = _congestion_eval_inst->evalRouteCong();
  // result.push_back(static_cast<float>(wirelength));
  // return result;
}

vector<float> EvalAPI::getUseCapRatioList()
{
  return _congestion_eval_inst->getUseCapRatioList();
}

vector<int> EvalAPI::getTileGridCoordSizeCntXY()
{
  vector<int> result_list;
  TileGrid* tile_grid = _congestion_eval_inst->get_tile_grid();
  result_list.emplace_back(tile_grid->get_lx());
  result_list.emplace_back(tile_grid->get_ly());
  result_list.emplace_back(tile_grid->get_tile_size_x());
  result_list.emplace_back(tile_grid->get_tile_size_y());
  result_list.emplace_back(tile_grid->get_tile_cnt_x());
  result_list.emplace_back(tile_grid->get_tile_cnt_y());
  return result_list;
}

void EvalAPI::plotGRCong(const string& plot_path, const string& output_file_name)
{
  _congestion_eval_inst->plotGRCong(plot_path, output_file_name);
}

void EvalAPI::plotOverflow(const string& plot_path, const string& output_file_name)
{
  _congestion_eval_inst->plotOverflow(plot_path, output_file_name);
}

void EvalAPI::reportCongestion(const string& plot_path, const string& output_file_name, const vector<CongNet*>& net_list, CongGrid* grid,
                               const vector<CongInst*>& inst_list)
{
  CongestionEval congestion_eval;
  congestion_eval.set_cong_grid(grid);
  congestion_eval.set_cong_inst_list(inst_list);
  congestion_eval.set_cong_net_list(net_list);
  congestion_eval.mapInst2Bin();
  congestion_eval.mapNetCoord2Grid();
  congestion_eval.reportCongestion(plot_path, output_file_name);
}
/******************************Congestion Eval: END******************************/

/****************************** Timing Eval: START ******************************/
void EvalAPI::initTimingDataFromIDB()
{
  _timing_eval_inst->initTimingDataFromIDB();
}

void EvalAPI::initTimingEval(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, vector<const char*> lib_file_path_list,
                             const char* sdc_file_path)
{
  _timing_eval_inst = new TimingEval();
  _timing_eval_inst->initTimingEngine(idb_builder, sta_workspace_path, lib_file_path_list, sdc_file_path);
}

void EvalAPI::initTimingEval(int32_t unit)
{
  _timing_eval_inst = new TimingEval();
  _timing_eval_inst->initTimingEngine(unit);
}

double EvalAPI::getEarlySlack(const string& pin_name) const
{
  return _timing_eval_inst->getEarlySlack(pin_name);
}

double EvalAPI::getLateSlack(const string& pin_name) const
{
  return _timing_eval_inst->getLateSlack(pin_name);
}

double EvalAPI::getArrivalEarlyTime(const string& pin_name) const
{
  return _timing_eval_inst->getArrivalEarlyTime(pin_name);
}

double EvalAPI::getArrivalLateTime(const string& pin_name) const
{
  return _timing_eval_inst->getArrivalLateTime(pin_name);
}

double EvalAPI::getRequiredEarlyTime(const string& pin_name) const
{
  return _timing_eval_inst->getRequiredEarlyTime(pin_name);
}

double EvalAPI::getRequiredLateTime(const string& pin_name) const
{
  return _timing_eval_inst->getRequiredLateTime(pin_name);
}

double EvalAPI::reportWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _timing_eval_inst->reportWNS(clock_name, mode);
  // if (_timing_eval_inst->checkClockName(clock_name)) {
  // }
}

double EvalAPI::reportTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _timing_eval_inst->reportTNS(clock_name, mode);
  // if (_timing_eval_inst->checkClockName(clock_name)) {
  // }
}

void EvalAPI::updateTiming(const vector<TimingNet*>& timing_net_list)
{
  _timing_eval_inst->updateEstimateDelay(timing_net_list);
}

void EvalAPI::updateTiming(const vector<TimingNet*>& timing_net_list, const vector<string>& name_list, const int& propagation_level)
{
  _timing_eval_inst->updateEstimateDelay(timing_net_list, name_list, propagation_level);
}

void EvalAPI::destroyTimingEval()
{
  delete _timing_eval_inst;
  _timing_eval_inst = nullptr;
}
/****************************** Timing Eval: END *******************************/
}  // namespace eval
