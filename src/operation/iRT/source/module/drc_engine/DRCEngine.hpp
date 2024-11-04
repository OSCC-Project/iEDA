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
#include "DEFuncType.hpp"
#include "DETask.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Monitor.hpp"

namespace irt {

#define RTDE (irt::DRCEngine::getInst())

class DRCEngine
{
 public:
  static void initInst();
  static DRCEngine& getInst();
  static void destroyInst();
  // function
  std::vector<Violation> getViolationList(DETask& de_task);

 private:
  // self
  static DRCEngine* _de_instance;

  DRCEngine() = default;
  DRCEngine(const DRCEngine& other) = delete;
  DRCEngine(DRCEngine&& other) = delete;
  ~DRCEngine() = default;
  DRCEngine& operator=(const DRCEngine& other) = delete;
  DRCEngine& operator=(DRCEngine&& other) = delete;
  // function
  void getViolationListBySelf(DETask& de_task);
  void buildTask(DETask& de_task);
  void writeTask(DETask& de_task);
  void readTask(DETask& de_task);
  void getViolationListByOther(DETask& de_task);
  void buildViolationList(DETask& de_task);
  DEProcessType getDEProcessType(Violation& violation);
  std::vector<Violation> expandViolation(Violation& violation);
  void buildByFunc(Violation& violation, const DEFuncType& de_func_type, DEProcessType& de_process_type,
                   std::vector<Violation>& expanded_violation_list);
  PlanarRect keepRect(Violation& violation);
  PlanarRect enlargeRect(Violation& violation);
  std::vector<std::pair<int32_t, bool>> keepLayer(Violation& violation);
  std::vector<std::pair<int32_t, bool>> expandOneViaLayer(Violation& violation);
  std::vector<std::pair<int32_t, bool>> expandTwoViaLayer(Violation& violation);
};

}  // namespace irt
