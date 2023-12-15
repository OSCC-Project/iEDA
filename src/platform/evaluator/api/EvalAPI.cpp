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

#include <iostream>

#include "PMAPI.hpp"
#include "RTAPI.hpp"
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

void EvalAPI::reportWirelength(const string& plot_path, const string& output_file_name, const vector<WLNet*>& net_list)
{
  WirelengthEval wirelength_eval;
  wirelength_eval.set_net_list(net_list);
  wirelength_eval.reportWirelength(plot_path, output_file_name);
}
/******************************Wirelength Eval: END******************************/

/******************************Congestion Eval: START******************************/

vector<float> EvalAPI::evalPinDens()
{
  _congestion_eval_inst->mapInst2Bin();
  return _congestion_eval_inst->evalPinDens();
}

vector<float> EvalAPI::evalPinDens(CongGrid* grid, const vector<CongInst*>& inst_list)
{
  CongestionEval congestion_eval;
  congestion_eval.set_cong_grid(grid);
  congestion_eval.set_cong_inst_list(inst_list);
  congestion_eval.mapInst2Bin();
  return congestion_eval.evalPinDens();
}

vector<float> EvalAPI::evalInstDens()
{
  _congestion_eval_inst->mapInst2Bin();
  return _congestion_eval_inst->getInstDens();
}

vector<float> EvalAPI::evalInstDens(CongGrid* grid, const vector<CongInst*>& inst_list)
{
  CongestionEval congestion_eval;
  congestion_eval.set_cong_grid(grid);
  congestion_eval.set_cong_inst_list(inst_list);
  congestion_eval.mapInst2Bin();
  return congestion_eval.getInstDens();
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
  // call router to get tilegrid info
  std::map<std::string, std::any> config_map;
  TileGrid* tile_grid = RTAPIInst.getCongestonMap(config_map);

  _congestion_eval_inst->set_tile_grid(tile_grid);
  return _congestion_eval_inst->evalRouteCong();
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

/****************************** GDS Wrapper: START ******************************/
vector<GDSNet*>& EvalAPI::wrapGDSNetlist(const string& eval_json)
{
  Config* config = Config::getOrCreateConfig(eval_json);
  Manager::initInst(config);
  auto gds_wrapper = Manager::getInst().getGDSWrapper();
  auto& gds_net_list = gds_wrapper->get_net_list();
  Manager::destroyInst();
  return gds_net_list;
}
/****************************** GDS Wrapper: END ********************************/

}  // namespace eval
