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

#include "ChangeType.hpp"
#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "IRModel.hpp"
#include "RTU.hpp"
#include "flute3/flute.h"

namespace irt {

#define IR_INST (irt::InitialRouter::getInst())

class InitialRouter
{
 public:
  static void initInst();
  static InitialRouter& getInst();
  static void destroyInst();
  // function
  void route(std::vector<Net>& net_list);

 private:
  // self
  static InitialRouter* _ir_instance;

  InitialRouter() { Flute::readLUT(); }
  InitialRouter(const InitialRouter& other) = delete;
  InitialRouter(InitialRouter&& other) = delete;
  ~InitialRouter() = default;
  InitialRouter& operator=(const InitialRouter& other) = delete;
  InitialRouter& operator=(InitialRouter&& other) = delete;
  // function
  IRModel initIRModel(std::vector<Net>& net_list);
  std::vector<IRNet> convertToIRNetList(std::vector<Net>& net_list);
  IRNet convertToIRNet(Net& net);
  void setIRParameter(IRModel& ir_model);
  void initLayerNodeMap(IRModel& ir_model);
  void buildIRNodeNeighbor(IRModel& ir_model);
  void buildAccessSupply(IRModel& ir_model);
  void checkIRModel(IRModel& ir_model);
  void routeIRModel(IRModel& ir_model);
  void updateIRModel(IRModel& ir_model);
};

}  // namespace irt
