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
#pragma once

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Monitor.hpp"
#include "VRModel.hpp"

namespace irt {

#define RTVR (irt::ViolationReporter::getInst())

class ViolationReporter
{
 public:
  static void initInst();
  static ViolationReporter& getInst();
  static void destroyInst();
  // function
  void report();

 private:
  // self
  static ViolationReporter* _vr_instance;

  ViolationReporter() = default;
  ViolationReporter(const ViolationReporter& other) = delete;
  ViolationReporter(ViolationReporter&& other) = delete;
  ~ViolationReporter() = default;
  ViolationReporter& operator=(const ViolationReporter& other) = delete;
  ViolationReporter& operator=(ViolationReporter&& other) = delete;
  // function
  VRModel initVRModel();
  std::vector<VRNet> convertToVRNetList(std::vector<Net>& net_list);
  VRNet convertToVRNet(Net& net);
  void uploadViolation(VRModel& vr_model);
  std::vector<Violation> getViolationList(VRModel& vr_model);

#if 1  // exhibit
  void updateSummary(VRModel& vr_model);
  void printSummary(VRModel& vr_model);
  void outputNetCSV(VRModel& vr_model);
  void outputViolationCSV(VRModel& vr_model);
  void outputNetJson(VRModel& vr_model);
  void outputViolationJson(VRModel& vr_model);
#endif

#if 1  // debug
  void debugPlotVRModel(VRModel& vr_model, std::string flag);
#endif
};

}  // namespace irt
