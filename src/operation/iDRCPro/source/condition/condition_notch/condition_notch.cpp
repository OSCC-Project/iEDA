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

#include "condition_notch.h"

#include "condition.h"
#include "idrc_util.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_condition_edge.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionNotch::checkFastMode()
{
  bool b_result = true;

  b_result &= checkNotch();

  return b_result;
}

bool DrcRuleConditionNotch::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkNotch();

  return b_result;
}

bool DrcRuleConditionNotch::checkNotch()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kEdgeNotch);
  for (auto& [layer, check_list] : check_map) {
    // get rule notch map
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kEdge);
    auto& rule_notch_map = rule_map->get_rule_map(RuleType::kEdgeNotch);

    for (auto& point_pair : check_list->get_points()) {
      // swap edge points order
      auto* point_1_next = point_pair.first->nextEndpoint();
      auto* point_2_next = point_pair.second->nextEndpoint();
      if (point_1_next != point_pair.second && point_2_next != point_pair.first) {
        continue;
      } else if (point_2_next == point_pair.first) {
        std::swap(point_pair.first, point_pair.second);
      }

      if (point_pair.first->is_notch_checked() || point_pair.second->is_notch_checked()) {
        continue;
      }

      // redundancy
      if (!checkNotch(point_pair.first, point_pair.second, layer, rule_notch_map)) {
        b_result = false;
      }
    }
  }
  return b_result;
}

bool DrcRuleConditionNotch::checkNotch(DrcBasicPoint* point_prev, DrcBasicPoint* point_next, idb::IdbLayer* layer,
                                       std::map<int, std::vector<ConditionRule*>> rule_notch_map)
{
  bool b_result = true;

  int notch_edge_length = point_prev->distance(point_next);
  auto* point_prev_prev = point_prev->prevEndpoint();                                  //
  auto* point_next_next = point_next->nextEndpoint();                                  //
  int notch_prev_side_length = point_prev->distance(point_prev_prev);                  //
  int notch_next_side_length = point_next->distance(point_next_next);                  //
  bool is_prev_concave = point_prev_prev->getCornerType() == DrcCornerType::kConcave;  // TODO：has Concave
  bool is_next_concave = point_next_next->getCornerType() == DrcCornerType::kConcave;
  int notch_prev_side_width = point_prev->distance(point_prev->get_neighbour(point_next->direction(point_prev))->get_point());
  int notch_next_side_width = point_next->distance(point_next->get_neighbour(point_prev->direction(point_next))->get_point());

  point_prev->set_checked_notch();
  point_next->set_checked_notch();
  point_prev_prev->set_checked_notch();
  point_next_next->set_checked_notch();

  // bool is_next_concave ? prevEndpoint->getCornerType();  //

  // find rule and check
  for (auto& [value, rule_notch_list] : rule_notch_map) {
    if (value <= notch_edge_length) {  // different rule
      continue;
    }

    for (auto& rule_notch : rule_notch_list) {
      // get rule data
      auto* condition_rule_notch = static_cast<ConditionRuleNotch*>(rule_notch);
      int notch_side_length = condition_rule_notch->get_notch()->get_min_notch_length();
      // optional
      int notch_side_width = condition_rule_notch->get_notch()->get_concave_ends_side_of_notch_width().has_value()
                                 ? condition_rule_notch->get_notch()->get_concave_ends_side_of_notch_width().value()
                                 : -1;

      bool is_violation = false;

      if ((notch_side_width < 0
           || (is_prev_concave && notch_prev_side_width <= notch_side_width && notch_next_side_width <= notch_side_width))
          && notch_prev_side_length < notch_side_length && notch_next_side_length > notch_side_length) {
        // violation rect
        int llx = point_next->get_x();
        int lly = point_next->get_y();
        int urx = point_prev->prevEndpoint()->get_x();
        int ury = point_prev->prevEndpoint()->get_y();
        std::set<int> net_ids;
        net_ids.insert(point_prev->get_id());

        is_violation = true;
#ifdef DEBUG_IDRC_CONDITION_NOTCH
        auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
        auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
#endif

        DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
        auto violation_type = ViolationEnumType::kViolationNotch;
        auto* violation_manager = _condition_manager->get_violation_manager();
        auto& violation_list = violation_manager->get_violation_list(violation_type);
        violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
      }

      else if ((notch_side_width < 0
                || (is_next_concave && notch_next_side_width <= notch_side_width && notch_prev_side_width <= notch_side_width))
               && notch_next_side_length < notch_side_length && notch_prev_side_length > notch_side_length) {
        // violation rect
        int llx = point_prev->get_x();
        int lly = point_prev->get_y();
        int urx = point_next->nextEndpoint()->get_x();
        int ury = point_next->nextEndpoint()->get_y();
        std::set<int> net_ids;
        net_ids.insert(point_prev->get_id());

        is_violation = true;
#ifdef DEBUG_IDRC_CONDITION_NOTCH
        auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
        auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
#endif

        DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
        auto violation_type = ViolationEnumType::kViolationNotch;
        auto* violation_manager = _condition_manager->get_violation_manager();
        auto& violation_list = violation_manager->get_violation_list(violation_type);
        violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
      }
    }
  }

  return b_result;
}

}  // namespace idrc