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
  auto gtl_pts_1 = DrcUtil::getPolygonPoints(check_region[0].second[1]);
  auto gtl_pts_2 = DrcUtil::getPolygonPoints(check_region[0].second[2]);
  auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
  auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());

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