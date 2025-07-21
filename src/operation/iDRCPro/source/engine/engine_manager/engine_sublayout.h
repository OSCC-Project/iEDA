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
#include <map>

#include "engine_geometry.h"
#include "geometry_boost.h"
#include <unordered_set>
namespace idrc {
class DrcEngineSubLayout
{
 public:
  DrcEngineSubLayout(int id);
  ~DrcEngineSubLayout();

  int get_id() { return _id; }
  // std::map<int, DrcEngineSubLayout*>& get_intersect_layouts() { return _intersect_layouts; }
  //   ieda_solver::EngineGeometry* get_engine() { return _engine; }
  ieda_solver::GeometryBoost* get_engine() { return (ieda_solver::GeometryBoost*) _engine; }
  bool isIntersect(int llx, int lly, int urx, int ury);
  void markChecked(int net_id);
  bool hasChecked(int net_id);
  bool clearChecked();

 private:
  /**
   * _id : net id
   * _engine : a geometry ptr including all shapes in one net
   */
  int _id = -1;
  std::unordered_set<int> _check_nets;  /// net ids in other sublayouts that has been checked with this sublayout
  ieda_solver::EngineGeometry* _engine = nullptr;
  // std::map<int, DrcEngineSubLayout*> _intersect_layouts;
};

}  // namespace idrc