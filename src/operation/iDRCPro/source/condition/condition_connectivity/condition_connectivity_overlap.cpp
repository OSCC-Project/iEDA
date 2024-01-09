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

#include "IdbLayer.h"
#include "condition.h"
#include "condition_connectivity.h"
#include "idrc_data.h"
#include "idrc_util.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_enum.h"

namespace idrc {
/**
 * check overlap for each layout(same layer) between different sub layout(different net)
 */
// bool DrcRuleConditionConnectivity::checkOverlap()
// {
//   bool b_result = true;

//   /// check routing layer
//   auto& engine_layouts = get_engine()->get_engine_manager()->get_engine_layouts(LayoutType::kRouting);
//   // layer id : indicate layer
//   // engine_layout : all nets shapes in the indicate layer
//   for (auto& [layer, engine_layout] : engine_layouts) {
//     /// sub_layouts : indicate shapes for all nets in one layer
//     auto& sub_layouts = engine_layout->get_sub_layouts();
//     /// no overlap in this layer
//     if ((int) sub_layouts.size() < MAX_CMP_NUM) {
//       continue;
//     }

//     /// compare polygons between different sub layout
//     /// iter_1 iter_2 : indicate shapes for one net in the same layer
//     for (auto iter_1 = sub_layouts.begin(); iter_1 != sub_layouts.end(); iter_1++) {
//       auto iter_2 = iter_1;
//       iter_2++;
//       for (; iter_2 != sub_layouts.end(); iter_2++) {
//         auto* engine_1 = iter_1->second->get_engine();
//         auto* engine_2 = iter_2->second->get_engine();
//         bool b_result_sub = engine_1->checkOverlap(engine_2);
//         if (b_result_sub == false) {
//           std::cout << "Check overlap layer_id = " << layer->get_id() << " net id 1= " << iter_1->first << " net id 2= " << iter_2->first
//                     << std::endl;
//         }

//         b_result &= b_result_sub;
//       }
//     }
//   }
//   return b_result;
// }

bool DrcRuleConditionConnectivity::checkOverlap()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kConnectivity);
  for (auto& [layer, check_list] : check_map) {
    // handle overlap data
    for (auto& point_pair : check_list->get_points()) {
      auto* segment = point_pair.first->get_neighbour(point_pair.first->direction(point_pair.second));
      if (!segment || !segment->get_type().hasType(ScanlineDataType::kEdge) || point_pair.first->is_overlap_checked()
          || point_pair.second->is_overlap_checked()) {
        continue;
      }

      // handle overlap segment
      if (!findOverlapRegion(point_pair.first, point_pair.second, layer)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionConnectivity::findOverlapRegion(DrcBasicPoint* point1, DrcBasicPoint* point2, idb::IdbLayer* layer)
{
  bool b_result = true;

  int llx = std::min(point1->get_x(), point2->get_x());
  int lly = std::min(point1->get_y(), point2->get_y());
  int urx = std::max(point1->get_x(), point2->get_x());
  int ury = std::max(point1->get_y(), point2->get_y());

  std::set<int> net_ids;
  net_ids.insert(point1->get_id());
  net_ids.insert(point2->get_id());

  std::function<void(DrcBasicPoint*)> find_connected_overlaps = [&](DrcBasicPoint* point) {
    if (point->is_overlap_checked()) {
      return;
    }

    point->set_checked_overlap();
    for (auto direction : {DrcDirection::kUp, DrcDirection::kDown, DrcDirection::kLeft, DrcDirection::kRight}) {
      auto* neighbour = point->get_neighbour(direction);
      if (neighbour && neighbour->is_overlap()) {
        llx = std::min(llx, neighbour->get_point()->get_x());
        lly = std::min(lly, neighbour->get_point()->get_y());
        urx = std::max(urx, neighbour->get_point()->get_x());
        ury = std::max(ury, neighbour->get_point()->get_y());

        net_ids.insert(neighbour->get_point()->get_id());
        if (neighbour->get_point()->get_id() != point1->get_id()) {
          point2 = neighbour->get_point();
        }

        find_connected_overlaps(neighbour->get_point());
      }
    }
  };

  find_connected_overlaps(point1);
  find_connected_overlaps(point2);

#if 1
  auto gtl_pts_1 = DrcUtil::getPolygonPoints(point1);
  auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
  auto gtl_pts_2 = DrcUtil::getPolygonPoints(point2);
  auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif

  // create violation
  DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
  auto violation_type = ViolationEnumType::kViolationShort;
  auto* violation_manager = _condition_manager->get_violation_manager();
  auto& violation_list = violation_manager->get_violation_list(violation_type);
  violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
  b_result = false;

  return b_result;
}

}  // namespace idrc