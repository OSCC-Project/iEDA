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

#define DEBUGCONDITION 1

#if DEBUGCONDITION
#define DEBUGPRINT 1
#define DEBUGCLOSE 0
#else
#define DEBUGPRINT 0
#define DEBUGCLOSE 0
#endif

#if DEBUGPRINT
#define DEBUGOUTPUT(x) (std::cout << "idrc : " << x << std::endl)
#define DEBUGHIGHLIGHT(x) "\033[0;36m" << x << "\033[0m"
#else
#define DEBUGOUTPUT(x)
#define DEBUGHIGHLIGHT(x)
#endif

#if DEBUGCLOSE
#if 1
// #define DEBUGCLOSE_OVERLAP
// #define DEBUGCLOSE_MINSPACING
// #define DEBUGCLOSE_JOG
// #define DEBUGCLOSE_PRL
#define DEBUGCLOSE_STEP
#define DEBUGCLOSE_HOLE
#define DEBUGCLOSE_AREA
// #define DEBUGCLOSE_EOL
// #define DEBUGCLOSE_CORNER_FILL
#define DEBUGCLOSE_NOTCH
#endif
#endif

class DrcEngineLayout;

class DrcConditionManager
{
 public:
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager) {}
  ~DrcConditionManager() {}

  DrcViolationManager* get_violation_manager() { return _violation_manager; }

  void set_check_select(std::set<ViolationEnumType> check_select)
  {
    if (check_select.empty()) {
      for (int type = (int) ViolationEnumType::kNone; type < (int) ViolationEnumType::kMax; ++type) {
        _check_select.insert((ViolationEnumType) type);
      }
    } else {
      _check_select = check_select;
    }
  }

  void checkOverlap(std::string layer, DrcEngineLayout* layout);
  void checkMinSpacing(std::string layer, DrcEngineLayout* layout);
  void checkWires(std::string layer, DrcEngineLayout* layout);
  void checkPolygons(std::string layer, DrcEngineLayout* layout);

 private:
  DrcViolationManager* _violation_manager;

  std::set<ViolationEnumType> _check_select;

  void addViolation(ieda_solver::GeometryRect& rect, std::string layer, ViolationEnumType type, std::set<int> net_id = {});
  void checkJog(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map);
  void checkSpacingTable(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& prl_wire_map);
};

}  // namespace idrc