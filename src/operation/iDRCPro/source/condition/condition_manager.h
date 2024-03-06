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

#include <list>
#include <map>
#include <string>
#include <vector>

#include "engine_geometry.h"
#include "idrc_util.h"
#include "idrc_violation_manager.h"
#include "tech_rules.h"

namespace idrc {

#define DEBUGPRINT 1
#if DEBUGPRINT
#define DEBUGOUTPUT(x) (std::cout << "idrc : " << x << std::endl)
#define DEBUGHIGHLIGHT(x) "\033[0;36m" << x << "\033[0m"
#else
#define DEBUGOUTPUT(x)
#define DEBUGHIGHLIGHT(x)
#endif

class DrcEngineLayout;

class DrcConditionManager
{
 public:
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager) {}
  ~DrcConditionManager() {}

  DrcViolationManager* get_violation_manager() { return _violation_manager; }

  void checkOverlap(std::string layer, DrcEngineLayout* layout);
  void checkMinSpacing(std::string layer, DrcEngineLayout* layout);
  void checkWires(std::string layer, DrcEngineLayout* layout);
  void checkPolygons(std::string layer, DrcEngineLayout* layout);

 private:
  DrcViolationManager* _violation_manager;

  void addViolation(ieda_solver::GeometryRect& rect, std::string layer, ViolationEnumType type, std::set<int> net_id = {});
  void checkJog(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map);
  void checkSpacingTable(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& prl_wire_map);
};

}  // namespace idrc