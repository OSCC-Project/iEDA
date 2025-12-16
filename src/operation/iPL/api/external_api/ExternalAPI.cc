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

#include <fstream>
#include <iostream>
#include <sstream>

#include "congestion_api.h"
#include "density_api.h"
#include "idm.h"
#include "report/ReportTable.hh"
#include "timing_api.hh"
#include "tool_api/ista_io/ista_io.h"
#include "wirelength_api.h"
#include "PLAPI.hh"

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

ieval::TotalWLSummary ExternalAPI::evalproIDBWL()
{
  return WIRELENGTH_API_INST->totalWL();
}

ieval::OverflowSummary ExternalAPI::evalproCongestion()
{
  iPLAPIInst.writeBackSourceDataBase();
  CONGESTION_API_INST->egrMap("place");

  return CONGESTION_API_INST->egrOverflow("place");
}

float ExternalAPI::obtainPeakAvgPinDens()
{
  ieval::PinMapSummary pin_map_summary = DENSITY_API_INST->pinDensityMap("place");

  std::string file_path = pin_map_summary.allcell_pin_density;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << file_path << std::endl;
    return -1;
  }

  std::string line;
  std::vector<float> values;
  float max_value = 0.f;
  float average = 0.f;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      float value = std::stof(cell);
      values.push_back(value);
      max_value = std::max(max_value, value);
    }
  }

  file.close();

  if (!values.empty()) {
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    average = sum / values.size();
  } else {
    average = 0.0f;
  }

  return max_value / average;
}

void ExternalAPI::destroyCongEval()
{
  ieval::CongestionAPI::destroyInst();
}

std::unique_ptr<ieda::ReportTable> ExternalAPI::generateTable(const std::string& name)
{
  return std::make_unique<ieda::ReportTable>(name.c_str());
}

}  // namespace ipl