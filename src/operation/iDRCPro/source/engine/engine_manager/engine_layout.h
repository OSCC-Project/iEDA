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
#include <stdint.h>

#include <map>

#include "boost_definition.h"
#include "engine_geometry.h"
#include "engine_sublayout.h"
#include "geometry_boost.h"

namespace idb {
class IdbLayer;
}  // namespace idb

namespace idrc {
/**
 *  DrcEngineLayout definition : describe all shapes for all nets in one layer
 */
typedef bg::index::rtree<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>, bg::index::quadratic<16>> EngineRTree;

class DrcDataManager;

class DrcEngineLayout
{
 public:
  DrcEngineLayout(std::string layer);
  ~DrcEngineLayout();

  std::map<int, DrcEngineSubLayout*>& get_sub_layouts() { return _sub_layouts; }
  DrcEngineSubLayout* get_sub_layout(int net_id);
  void clearSublayoutMark();  // clear all sub layout marked net
  //   ieda_solver::EngineGeometry* get_net_engine(int net_id);
  //   ieda_solver::EngineGeometry* get_layout_engine() { return _engine; }
  ieda_solver::GeometryBoost* get_net_engine(int net_id);
  ieda_solver::GeometryBoost* get_layout_engine() { return (ieda_solver::GeometryBoost*) _engine; }

  //   uint64_t pointCount();

  bool addRect(int llx, int lly, int urx, int ury, int net_id);

  void combineLayout();

  /// engine RTree
  void addRTreeSubLayout(DrcEngineSubLayout* sub_layout);
  std::vector<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>> querySubLayouts(int llx, int lly, int urx, int ury);
  std::set<int> querySubLayoutNetId(int llx, int lly, int urx, int ury);

 private:
  /**
   * _layer : layer name
   */
  std::string _layer;
  /**
   * int : net id, if equal to -1, sub layout is a kind of blockage
   * DrcEngineSubLayout* : sub layout ptr describe the net shapes
   */
  std::map<int, DrcEngineSubLayout*> _sub_layouts;
  /**
   * region query
   */
  EngineRTree _query_tree;

  /// whole design
  ieda_solver::EngineGeometry* _engine = nullptr;
};

}  // namespace idrc