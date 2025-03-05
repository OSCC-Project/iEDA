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
#include "idrc_data.h"
#include "idrc_util.h"
#include "idrc_violation_manager.h"
#include "tech_rules.h"

namespace idrc {

class DrcEngineLayout;

class DrcConditionManager
{
 public:
  DrcConditionManager(DrcViolationManager* violation_manager) : _violation_manager(violation_manager) {}
  ~DrcConditionManager() {}

  DrcViolationManager* get_violation_manager() { return _violation_manager; }

  void set_check_select(std::set<ViolationEnumType> check_select);
  void set_check_type(DrcCheckerType check_type);

  void checkOverlap(std::string layer, DrcEngineLayout* layout);
  void checkMinSpacing(std::string layer, DrcEngineLayout* layout);
  void checkArea(std::string layer, DrcEngineLayout* layout);

  void checkPolygons(std::string layer, DrcEngineLayout* layout);
  void checkParallelLengthSpacing(std::string layer, DrcEngineLayout* layout);
  void checkJogToJogSpacing(std::string layer, DrcEngineLayout* layout);

 private:
  DrcViolationManager* _violation_manager;
  DrcCheckerType _check_type;

  std::set<ViolationEnumType> _check_select;

  void checkOverlapByInteract(std::string layer, DrcEngineLayout* layout);
  void checkOverlapBySelfIntersect(std::string layer, DrcEngineLayout* layout);

  void addViolation(ieda_solver::GeometryRect& rect, std::string layer, ViolationEnumType type, std::set<int> net_id = {});
  void buildMapOfJog(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map);
  void checkJog(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map);
  void buildMapOfSpacingTable(std::string layer, DrcEngineLayout* layout,
                              std::map<int, std::vector<ieda_solver::GeometryRect>>& prl_wire_map,
                              std::map<int, std::vector<ieda_solver::GeometryPolygonSet>>& prl_polygon_map);
  void checkSpacingTable(std::string layer, DrcEngineLayout* layout, std::map<int, std::vector<ieda_solver::GeometryRect>>& prl_wire_map,
                         std::map<int, std::vector<ieda_solver::GeometryPolygonSet>>& prl_polygon_map);
};

}  // namespace idrc