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
  void init();
  std::vector<Violation> getViolationList(DETask& de_task);
  void addTempIgnoredViolation(std::vector<Violation>& violation_list);
  void clearTempIgnoredViolationSet();
  void destroy();

 private:
  // self
  static DRCEngine* _de_instance;
  std::set<Violation, CmpViolation> _ignored_violation_set;
  std::set<Violation, CmpViolation> _temp_ignored_violation_set;

  DRCEngine() = default;
  DRCEngine(const DRCEngine& other) = delete;
  DRCEngine(DRCEngine&& other) = delete;
  ~DRCEngine() = default;
  DRCEngine& operator=(const DRCEngine& other) = delete;
  DRCEngine& operator=(DRCEngine&& other) = delete;
  // function
  void buildIgnoredViolationSet();
  void getViolationListByInterface(DETask& de_task);
  void filterViolationList(DETask& de_task);
  void checkViolationList(DETask& de_task);
  void buildViolationList(DETask& de_task);

#if 1  // aux
  bool skipViolation(DETask& de_task, Violation& violation);
  std::vector<Violation> getExpandedViolationList(DETask& de_task, Violation& violation);
  PlanarRect enlargeRect(PlanarRect& real_rect, int32_t required_size);
  std::vector<std::pair<int32_t, bool>> expandLayer(Violation& violation, std::vector<int32_t> offset_list);
#endif
};

}  // namespace irt
