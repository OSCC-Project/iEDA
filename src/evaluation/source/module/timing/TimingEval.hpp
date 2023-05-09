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
#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_MODULE_TIMING_TIMINGEVAL_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_MODULE_TIMING_TIMINGEVAL_HPP_

#include "TimingNet.hpp"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"
#include "netlist/Net.hh"

namespace eval {

class TimingEval
{
 public:
  TimingEval() = default;
  TimingEval(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
             const char* sdc_file_path);
  ~TimingEval() = default;

  // getter, called by iPL
  double getEarlySlack(const std::string& pin_name) const;
  double getLateSlack(const std::string& pin_name) const;
  double getArrivalEarlyTime(const std::string& pin_name) const;
  double getArrivalLateTime(const std::string& pin_name) const;
  double getRequiredEarlyTime(const std::string& pin_name) const;
  double getRequiredLateTime(const std::string& pin_name) const;
  double reportWNS(const char* clock_name, ista::AnalysisMode mode);
  double reportTNS(const char* clock_name, ista::AnalysisMode mode);

  // getter, called by iRT
  double getNetDelay(const char* net_name, const char* load_pin_name, ista::AnalysisMode mode, ista::TransType trans_type);

  // setter
  void set_timing_net_list(const std::vector<TimingNet*>& timing_net_list) { _timing_net_list = timing_net_list; }

  // adder
  void add_timing_net(TimingNet* timing_net) { _timing_net_list.push_back(timing_net); }
  TimingNet* add_timing_net(const std::string& name);
  void add_timing_net(const std::string& name, const std::vector<std::pair<TimingPin*, TimingPin*>>& pin_pair_list);

  // timing evaluator
  void estimateDelay(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
                     const char* sdc_file_path);
  void estimateDelay(std::vector<std::string> lef_file_path_list, std::string def_file_path, const char* sta_workspace_path,
                     std::vector<const char*> lib_file_path_list, const char* sdc_file_path);
  void updateEstimateDelay(const std::vector<TimingNet*>& timing_net_list);
  void updateEstimateDelay(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                           int propagation_level);

  // init timing_engine
  void initTimingEngine(int32_t unit);
  void initTimingEngine(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
                        const char* sdc_file_path);

  bool checkClockName(const char* clock_name);
  std::vector<const char*> getClockNameList();

 private:
  std::vector<TimingNet*> _timing_net_list;
  ista::TimingEngine* _timing_engine = nullptr;
  int32_t _unit = -1;
};
}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_MODULE_TIMING_TIMINGEVAL_HPP_
