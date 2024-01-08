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

#include "ExternalAPI.hh"

#include "EvalAPI.hpp"
#include "report/ReportTable.hh"
#include "timing/TimingEval.hpp"
#include "tool_api/ista_io/ista_io.h"

namespace ipl {

bool ExternalAPI::isSTAStarted()
{
  return staInst->isInitSTA();
}

void ExternalAPI::initSTA()
{
  staInst->initSTA();
  staInst->buildGraph();
}

void ExternalAPI::initEval()
{
  EvalInst.initInst();
}

void ExternalAPI::updateSTATiming()
{
  staInst->updateTiming();
}

bool ExternalAPI::isClockNet(std::string net_name)
{
  bool flag = staInst->isClockNet(net_name);
  return flag;
}

bool ExternalAPI::isSequentialCell(std::string inst_name)
{
  return staInst->isSequentialCell(inst_name);
}

bool ExternalAPI::isBufferCell(std::string cell_name)
{
  std::string cell_type = staInst->getCellType(cell_name.c_str());
  bool flag = (cell_type == "Buffer");
  return flag;
}

std::vector<std::string> ExternalAPI::obtainClockNameList()
{
  return staInst->getClockNameList();
}

bool ExternalAPI::insertSignalBuffer(std::pair<std::string, std::string> source_sink_net, std::vector<std::string> sink_pin_list,
                                     std::pair<std::string, std::string> master_inst_buffer, std::pair<int, int> buffer_center_loc)
{
  bool flag = staInst->insertBuffer(source_sink_net, sink_pin_list, master_inst_buffer, buffer_center_loc, idb::IdbConnectType::kSignal);
  return flag;
}

void ExternalAPI::initTimingEval(int32_t unit)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initTimingEval(unit);
}

double ExternalAPI::obtainPinEarlySlack(std::string pin_name)
{
  return eval::EvalAPI::getInst().getEarlySlack(pin_name);
}

double ExternalAPI::obtainPinLateSlack(std::string pin_name)
{
  return eval::EvalAPI::getInst().getLateSlack(pin_name);
}

double ExternalAPI::obtainPinEarlyArrivalTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getArrivalEarlyTime(pin_name);
}

double ExternalAPI::obtainPinLateArrivalTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getArrivalLateTime(pin_name);
}

double ExternalAPI::obtainPinEarlyRequiredTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getRequiredEarlyTime(pin_name);
}

double ExternalAPI::obtainPinLateRequiredTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getRequiredLateTime(pin_name);
}

double ExternalAPI::obtainWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return eval::EvalAPI::getInst().reportWNS(clock_name, mode);
}

double ExternalAPI::obtainTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return eval::EvalAPI::getInst().reportTNS(clock_name, mode);
}

void ExternalAPI::updateEvalTiming(const std::vector<eval::TimingNet*>& timing_net_list)
{
  EvalInst.updateTiming(timing_net_list);
}

void ExternalAPI::updateEvalTiming(const std::vector<eval::TimingNet*>& timing_net_list, const std::vector<std::string>& name_list, const int& propagation_level){
  EvalInst.updateTiming(timing_net_list,name_list,propagation_level);
}

float ExternalAPI::obtainInstPinCap(std::string inst_pin_name){
  return staInst->obtainInstPinCap(inst_pin_name);
}

float ExternalAPI::obtainAvgWireResUnitLengthUm(){
  return staInst->obtainAvgWireResUnitLengthUm();
}

float ExternalAPI::obtainAvgWireCapUnitLengthUm(){
  return staInst->obtainAvgWireCapUnitLengthUm();
}

float ExternalAPI::obtainInstOutPinRes(std::string cell_name, std::string port_name){
  return staInst->obtainInstOutPinRes(cell_name, port_name);
}

void ExternalAPI::destroyTimingEval()
{
  eval::EvalAPI::destroyInst();
}

/**
 * @brief run GR based on dmInst data, evaluate 3D congestion, and return <ACE,TOF,MOF> vector
 * @return std::vector<float>
 */
std::vector<float> ExternalAPI::evalGRCong()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();

  std::vector<float> gr_congestion;
  gr_congestion = eval_api.evalGRCong();

  return gr_congestion;
}

/**
 * @brief compute each gcellgrid routing demand/resource, and return a 2D route util map
 * @return std::vector<float>
 */
std::vector<float> ExternalAPI::getUseCapRatioList()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  return eval_api.getUseCapRatioList();
}

/**
 * @brief draw congesiton map based on GR result
 * @param  plot_path
 * @param  output_file_name
 */
void ExternalAPI::plotCongMap(const std::string& plot_path, const std::string& output_file_name)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  // layer by layer
  eval_api.plotGRCong(plot_path, output_file_name);
  // statistical TotalOverflow/MaximumOverflow
  eval_api.plotOverflow(plot_path, output_file_name);
}

void ExternalAPI::destroyCongEval()
{
  eval::EvalAPI::destroyInst();
}

std::vector<float> ExternalAPI::obtainPinDens()
{
  return eval::EvalAPI::getInst().evalPinDens();
}

std::vector<float> ExternalAPI::obtainNetCong(std::string rudy_type)
{
  return eval::EvalAPI::getInst().evalNetCong(rudy_type);
}

std::unique_ptr<ieda::ReportTable> ExternalAPI::generateTable(const std::string& name)
{
  return std::make_unique<ieda::ReportTable>(name.c_str());
}

}  // namespace ipl