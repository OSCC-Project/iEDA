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
#include "Database.hpp"
#include "Net.hpp"
#include "PADataManager.hpp"
#include "PAModel.hpp"
#include "PANode.hpp"

namespace irt {

#define PA_INST (irt::PinAccessor::getInst())

class PinAccessor
{
 public:
  static void initInst(Config& config, Database& database);
  static PinAccessor& getInst();
  static void destroyInst();
  // function
  void access(std::vector<Net>& net_list);

 private:
  // self
  static PinAccessor* _pa_instance;
  // config & database
  PADataManager _pa_data_manager;

  PinAccessor(Config& config, Database& database) { init(config, database); }
  PinAccessor(const PinAccessor& other) = delete;
  PinAccessor(PinAccessor&& other) = delete;
  ~PinAccessor() = default;
  PinAccessor& operator=(const PinAccessor& other) = delete;
  PinAccessor& operator=(PinAccessor&& other) = delete;
  // function
  void init(Config& config, Database& database);
  void accessPANetList(std::vector<PANet>& pa_net_list);

#if 1  // build pa_model
  PAModel initPAModel(std::vector<PANet>& pa_net_list);
  void buildPAModel(PAModel& pa_model);
  void initGCellRealRect(PAModel& pa_model);
  void addBlockageList(PAModel& pa_model);
  void cutBlockageList(PAModel& pa_model);
#endif

#if 1  // access pa_model
  void accessPAModel(PAModel& pa_model);
  void accessPANet(PAModel& pa_model, PANet& pa_net);
  void initAccessPointList(PANet& pa_net);
  std::vector<LayerRect> getIntersectPinShapeList(PAPin& pa_pin);
  void mergeAccessPointList(PANet& pa_net);
  void selectAccessPointList(PANet& pa_net);
#endif

#if 1  // update pa_model
  void updatePAModel(PAModel& pa_model);
  void buildBoundingBox(PANet& pa_net);
  void buildAccessPointList(PANet& pa_net);
  void buildDrivingPin(PANet& pa_net);
  void updateOriginPAResult(PAModel& pa_model);
#endif

#if 1  // check pa_model
  void checkPAModel(PAModel& pa_model);
#endif

#if 1  // report pa_model
  void countPAModel(PAModel& pa_model);
  void reportPAModel(PAModel& pa_model);
#endif
};

}  // namespace irt
