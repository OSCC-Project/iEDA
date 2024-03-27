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
#include "engine_geometry.h"

namespace idrc {
class DrcDataManager;

class DrcRegionQuery
{
 public:
  DrcRegionQuery(DrcDataManager* data_manager) : _data_manager(data_manager) {}
  ~DrcRegionQuery() { _data_manager = nullptr; }

  std::set<int> queryNetId(std::string layer, int llx, int lly, int urx, int ury)
  {
    std::set<int> net_ids;
    std::vector<std::pair<ieda_solver::BgRect, int>> result;
    ieda_solver::BgRect rect(ieda_solver::BgPoint(llx, lly), ieda_solver::BgPoint(urx, ury));
    _query_tree[layer].query(bg::index::intersects(rect), std::back_inserter(result));
    for (auto& pair : result) {
      net_ids.insert(pair.second);
    }
    return net_ids;
  }

  void addRect(ieda_solver::GeometryRect rect, std::string layer, int id)
  {
    ieda_solver::BgRect rtree_rect(ieda_solver::BgPoint(ieda_solver::lowLeftX(rect), ieda_solver::lowLeftY(rect)),
                                   ieda_solver::BgPoint(ieda_solver::upRightX(rect), ieda_solver::upRightY(rect)));
    _query_tree[layer].insert(std::make_pair(rtree_rect, id));
  }

 private:
  DrcDataManager* _data_manager = nullptr;

  std::map<std::string, bg::index::rtree<std::pair<ieda_solver::BgRect, int>, bg::index::quadratic<16>>> _query_tree;
};

}  // namespace idrc