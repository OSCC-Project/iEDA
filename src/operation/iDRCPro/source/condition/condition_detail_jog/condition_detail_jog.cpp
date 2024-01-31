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
#include "condition_detail_jog.h"

#include "idrc_util.h"

namespace idrc {

bool ConditionDetailJog::apply(std::vector<std::pair<ConditionSequence::SequenceType, std::vector<DrcBasicPoint*>>>& check_region)
{
  // TODO: jog check

  std::vector<DrcBasicPoint*> points;
  for (auto line : check_region) {
    std::copy(line.second.begin(), line.second.end(), std::back_inserter(points));
  }
  std::vector<ieda_solver::GtlPolygon> polygons;
  std::set<int> polygon_ids;
  for (auto* point : points) {
    if (polygon_ids.find(point->get_polygon_id()) == polygon_ids.end()) {
      polygon_ids.insert(point->get_polygon_id());
      auto gtl_pts = DrcUtil::getPolygonPoints(point);
      polygons.emplace_back(gtl_pts.begin(), gtl_pts.end());
    }
  }
  if (check_region[0].second[1]->get_y() == check_region[0].second[2]->get_y()) {
    int a = 0;
  }

  // std::vector<ieda_solver::GtlPoint> spacing_up_points;
  // std::vector<ieda_solver::GtlPoint> spacing_down_points;
  // for (auto& line : check_region) {
  //   if (spacing_up_points.empty() || line.second[2]->get_x() != spacing_up_points.back().x()
  //       || line.second[2]->get_y() != spacing_up_points.back().y()) {
  //     spacing_up_points.emplace_back(line.second[2]->get_x(), line.second[2]->get_y());
  //   }
  //   if (spacing_down_points.empty() || line.second[1]->get_x() != spacing_down_points.back().x()
  //       || line.second[1]->get_y() != spacing_down_points.back().y()) {
  //     spacing_down_points.emplace_back(line.second[1]->get_x(), line.second[1]->get_y());
  //   }
  // }
  // std::reverse(spacing_down_points.begin(), spacing_down_points.end());
  // spacing_up_points.insert(spacing_up_points.end(), spacing_down_points.begin(), spacing_down_points.end());
  // auto spacing_region = ieda_solver::GtlPolygon(spacing_up_points.begin(), spacing_up_points.end());

  return false;
}

// bool ConditionDetailJog::apply(CheckItem* item)
// {
//   auto* jog_item = dynamic_cast<ConditionJogCheckItem*>(item);
//   if (jog_item == nullptr) {
//     return false;
//   }
//   // TODO: jog check
//   return true;
// }

}  // namespace idrc