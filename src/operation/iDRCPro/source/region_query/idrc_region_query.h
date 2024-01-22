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

#include "boost_definition.h"
#include "drc_basic_point.h"

namespace idrc {
class DrcDataManager;

class DrcRegionQuery
{
 public:
  DrcRegionQuery(DrcDataManager* data_manager) : _data_manager(data_manager) {}
  ~DrcRegionQuery() { _data_manager = nullptr; }

  std::vector<std::pair<DrcBasicPoint*, DrcBasicPoint*>> getEdgesInRect(int llx, int lly, int urx, int ury)
  {
    std::vector<std::pair<DrcBasicPoint*, DrcBasicPoint*>> segments;
    std::vector<std::pair<ieda_solver::BgSegment, std::pair<DrcBasicPoint*, DrcBasicPoint*>>> result;
    ieda_solver::BgRect rect(ieda_solver::BgPoint(llx, lly), ieda_solver::BgPoint(urx, ury));
    _polygon_edge_rtree.query(bg::index::intersects(rect), std::back_inserter(result));
    for (auto& pair : result) {
      segments.emplace_back(pair.second);
    }
    return segments;
  }

  void addEdge(std::pair<DrcBasicPoint*, DrcBasicPoint*> edge)
  {
    _polygon_edge_rtree.insert(std::make_pair(ieda_solver::BgSegment(ieda_solver::BgPoint(edge.first->get_x(), edge.first->get_y()),
                                                                     ieda_solver::BgPoint(edge.second->get_x(), edge.second->get_y())),
                                              edge));
  }

 private:
  DrcDataManager* _data_manager = nullptr;

  bg::index::rtree<std::pair<ieda_solver::BgSegment, std::pair<DrcBasicPoint*, DrcBasicPoint*>>, bg::index::quadratic<16>>
      _polygon_edge_rtree;
};

}  // namespace idrc