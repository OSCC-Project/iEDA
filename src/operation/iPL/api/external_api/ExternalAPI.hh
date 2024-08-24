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

#ifndef IPL_EXTERNAL_API_H
#define IPL_EXTERNAL_API_H

#include <memory>

#include "ids.hh"

namespace ipl {

class ExternalAPI
{
 public:
  ExternalAPI() = default;
  ExternalAPI(const ExternalAPI&) = delete;
  ExternalAPI(ExternalAPI&&) = delete;
  ~ExternalAPI() = default;
  ExternalAPI& operator=(const ExternalAPI&) = delete;
  ExternalAPI& operator=(ExternalAPI&&) = delete;

  std::string obtainTargetDir();

  /*****************************Timing Interface: START*****************************/
  bool isSTAStarted();
  void modifySTAOutputDir(std::string path);
  void initSTA(std::string path, bool init_log);
  void initEval();
  void updateSTATiming();
  std::vector<std::string> obtainClockNameList();
  bool isClockNet(std::string net_name);
  bool isSequentialCell(std::string inst_name);
  bool isBufferCell(std::string cell_name);

  bool insertSignalBuffer(std::pair<std::string, std::string> source_sink_net, std::vector<std::string> sink_pin_list,
                          std::pair<std::string, std::string> master_inst_buffer, std::pair<int, int> buffer_center_loc);

  void initTimingEval(int32_t unit);
  double obtainPinEarlySlack(std::string pin_name);
  double obtainPinLateSlack(std::string pin_name);
  double obtainPinEarlyArrivalTime(std::string pin_name);
  double obtainPinLateArrivalTime(std::string pin_name);
  double obtainPinEarlyRequiredTime(std::string pin_name);
  double obtainPinLateRequiredTime(std::string pin_name);
  double obtainWNS(const char* clock_name, ista::AnalysisMode mode);
  double obtainTNS(const char* clock_name, ista::AnalysisMode mode);
  double obtainTargetClockPeriodNS(std::string clock_name);
  void updateEvalTiming(const std::vector<eval::TimingNet*>& timing_net_list);
  void updateEvalTiming(const std::vector<eval::TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                        const int& propagation_level);
  float obtainPinCap(std::string inst_pin_name);
  float obtainAvgWireResUnitLengthUm();
  float obtainAvgWireCapUnitLengthUm();
  float obtainInstOutPinRes(std::string cell_name, std::string port_name);
  void destroyTimingEval();
  /*****************************Timing Interface: END*******************************/

  /*****************************Routing Interface: START*****************************/
  std::vector<float> obtainPinDens(int32_t grid_cnt_x, int32_t grid_cnt_y);
  std::vector<float> obtainNetCong(std::string rudy_type);
  std::vector<float> evalGRCong();
  int64_t evalEGRWL();
  void evalproCongestion();
  int32_t evalprohpWL();
  int32_t evalproflute();

  std::vector<float> getUseCapRatioList();
  void plotCongMap(const std::string& plot_path, const std::string& output_file_name);
  void destroyCongEval();
  /*****************************Routing Interface: END*******************************/

  /*****************************Report Interface: START*****************************/
  std::unique_ptr<ieda::ReportTable> generateTable(const std::string& name);

  /*****************************Report Interface: END*******************************/
};

}  // namespace ipl

#endif