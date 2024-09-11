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

#include "idm.h"
#include "report/ReportTable.hh"
#include "congestion_api.h"
#include "timing_api.hh"
#include "tool_api/ista_io/ista_io.h"
#include "wirelength_api.h"

namespace ipl {

std::string ExternalAPI::obtainTargetDir()
{
  return dmInst->get_config().get_output_path();
}

bool ExternalAPI::isSTAStarted()
{
  return staInst->isInitSTA();
}

void ExternalAPI::modifySTAOutputDir(std::string path)
{
  staInst->setStaWorkDirectory(path);
}

void ExternalAPI::initSTA(std::string path, bool init_log)
{
  staInst->initSTA(path, init_log);
  staInst->buildGraph();
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

double ExternalAPI::obtainPinEarlySlack(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getEarlySlack(pin_name);
}

double ExternalAPI::obtainPinLateSlack(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getLateSlack(pin_name);
}

double ExternalAPI::obtainPinEarlyArrivalTime(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getArrivalEarlyTime(pin_name);
}

double ExternalAPI::obtainPinLateArrivalTime(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getArrivalLateTime(pin_name);
}

double ExternalAPI::obtainPinEarlyRequiredTime(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getRequiredEarlyTime(pin_name);
}

double ExternalAPI::obtainPinLateRequiredTime(std::string pin_name)
{
  return ieval::TimingAPI::getInst()->getRequiredLateTime(pin_name);
}

double ExternalAPI::obtainWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return ieval::TimingAPI::getInst()->reportWNS(clock_name, mode);
}

double ExternalAPI::obtainTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return ieval::TimingAPI::getInst()->reportTNS(clock_name, mode);
}

double ExternalAPI::obtainTargetClockPeriodNS(std::string clock_name)
{
  return staInst->getPeriodNS(clock_name);
}

void ExternalAPI::updateEvalTiming(const std::vector<ieval::TimingNet*>& timing_net_list, int32_t dbu_unit)
{
  ieval::TimingAPI::getInst()->updateTiming(timing_net_list, dbu_unit);
}

void ExternalAPI::updateEvalTiming(const std::vector<ieval::TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                                   const int& propagation_level, int32_t dbu_unit)
{
  ieval::TimingAPI::getInst()->updateTiming(timing_net_list, name_list, propagation_level, dbu_unit);
}

float ExternalAPI::obtainPinCap(std::string inst_pin_name)
{
  return staInst->obtainPinCap(inst_pin_name);
}

float ExternalAPI::obtainAvgWireResUnitLengthUm()
{
  return staInst->obtainAvgWireResUnitLengthUm();
}

float ExternalAPI::obtainAvgWireCapUnitLengthUm()
{
  return staInst->obtainAvgWireCapUnitLengthUm();
}

float ExternalAPI::obtainInstOutPinRes(std::string cell_name, std::string port_name)
{
  return staInst->obtainInstOutPinRes(cell_name, port_name);
}

void ExternalAPI::destroyTimingEval()
{
  ieval::TimingAPI::destroyInst();
}

/**
 * @brief run GR based on dmInst data, evaluate 3D congestion, and return <ACE,TOF,MOF> vector
 * @return std::vector<float>
 */
std::vector<float> ExternalAPI::evalGRCong()
{
  // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  // eval::EvalAPI& eval_api = EvalInst;
  // std::vector<float> gr_congestion;
  // gr_congestion = eval_api.evalGRCong();

  // eval::EvalAPI::destroyInst();

  // return gr_congestion;
  return {};
}

float ExternalAPI::evalproCongestion()
{
  // ieval::WirelengthAPI wirelength_api;
  // return wirelength_api.totalEGRWL("./rt_temp_directory/initial_router/route.guide");
  return 0.f;
}

ieval::TotalWLSummary ExternalAPI::evalproWL(std::vector<std::vector<std::pair<int32_t, int32_t>>> point_sets)
{
  // ieval::WirelengthAPI wirelength_api;
  // ieval::TotalWLSummary total_wl = wirelength_api.totalWL(point_sets);
  // return total_wl;
  ieval::TotalWLSummary total_wl;
  return total_wl;
}

int32_t ExternalAPI::evalproGRWL()
{
  // ieval::WirelengthAPI wirelength_api;
  // return wirelength_api.totalEGRWL("./rt_temp_directory/initial_router/route.guide");3
  return 0;
}

ieval::TotalWLSummary ExternalAPI::evalproIDBWL()
{
  // ieval::WirelengthAPI wirelength_api;
  // ieval::TotalWLSummary total_wl = wirelength_api.totalWL();
  // return total_wl;
  ieval::TotalWLSummary total_wl;
  return total_wl;
}

int32_t ExternalAPI::evalprohpWL()
{
  // ieval::WirelengthAPI wirelength_api;
  // std::vector<std::pair<int32_t, int32_t>> point_set;
  // std::pair<int32_t, int32_t> point1(0, 0);
  // std::pair<int32_t, int32_t> point2(3, 6);
  // std::pair<int32_t, int32_t> point3(4, 4);
  // std::pair<int32_t, int32_t> point4(6, 3);

  // point_set.push_back(point1);
  // point_set.push_back(point2);
  // point_set.push_back(point3);
  // point_set.push_back(point4);

  // std::vector<std::vector<std::pair<int32_t, int32_t>>> point_sets;
  // point_sets.push_back(point_set);

  // ieval::TotalWLSummary total_wl = wirelength_api.totalWL(point_sets);

  // return total_wl.HPWL;
  return 0;
}

int32_t ExternalAPI::evalproflute()
{
  // ieval::WirelengthAPI wirelength_api;
  // std::vector<std::pair<int32_t, int32_t>> point_set;
  // std::pair<int32_t, int32_t> point1(0, 0);
  // std::pair<int32_t, int32_t> point2(3, 6);
  // std::pair<int32_t, int32_t> point3(4, 4);
  // std::pair<int32_t, int32_t> point4(6, 3);

  // point_set.push_back(point1);
  // point_set.push_back(point2);
  // point_set.push_back(point3);
  // point_set.push_back(point4);

  // std::vector<std::vector<std::pair<int32_t, int32_t>>> point_sets;
  // point_sets.push_back(point_set);

  // ieval::TotalWLSummary total_wl = wirelength_api.totalWL(point_sets);

  // return total_wl.FLUTE;
  return 0;
}

int64_t ExternalAPI::evalEGRWL()
{
  // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();

  // int64_t egr_wl = static_cast<int64_t>(eval_api.evalEGRWL());

  // eval::EvalAPI::destroyInst();

  // return egr_wl;
  return 0;
}

/**
 * @brief compute each gcellgrid routing demand/resource, and return a 2D route util map
 * @return std::vector<float>
 */
std::vector<float> ExternalAPI::getUseCapRatioList()
{
  // eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  // return eval_api.getUseCapRatioList();
  return {};
}

/**
 * @brief draw congesiton map based on GR result
 * @param  plot_path
 * @param  output_file_name
 */
void ExternalAPI::plotCongMap(const std::string& plot_path, const std::string& output_file_name)
{
  // eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  // // layer by layer
  // eval_api.plotGRCong(plot_path, output_file_name);
  // // statistical TotalOverflow/MaximumOverflow
  // // eval_api.plotOverflow(plot_path, output_file_name);
}

void ExternalAPI::destroyCongEval()
{
  // eval::EvalAPI::destroyInst();
  ieval::CongestionAPI::destroyInst();
}

std::vector<float> ExternalAPI::obtainPinDens(int32_t grid_cnt_x, int32_t grid_cnt_y)
{
  // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  // int32_t bin_cnt_x = grid_cnt_x;
  // int32_t bin_cnt_y = grid_cnt_y;

  // eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  // std::vector<float> pin_num_list = eval_api.evalPinDens();

  // std::vector<float> result;

  // float sum = std::accumulate(pin_num_list.begin(), pin_num_list.end(), 0.0);
  // float average = sum / pin_num_list.size();
  // result.push_back(average);

  // auto max_element_ptr = std::max_element(pin_num_list.begin(), pin_num_list.end());

  // result.push_back((*max_element_ptr) / average);

  // eval::EvalAPI::destroyInst();
  // return result;
  return {};
}

std::vector<float> ExternalAPI::obtainNetCong(std::string rudy_type)
{
  // return eval::EvalAPI::getInst().evalNetCong(rudy_type);
  return {};
}

std::unique_ptr<ieda::ReportTable> ExternalAPI::generateTable(const std::string& name)
{
  return std::make_unique<ieda::ReportTable>(name.c_str());
}

}  // namespace ipl