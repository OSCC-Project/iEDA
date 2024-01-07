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

#include "condition_eol.h"

#include "condition.h"
#include "idrc_util.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_condition_edge.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionEOL::checkFastMode()
{
  bool b_result = true;

  b_result &= checkSpacingEOL();

  return b_result;
}

bool DrcRuleConditionEOL::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkSpacingEOL();

  return b_result;
}

bool DrcRuleConditionEOL::checkSpacingEOL()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kEdgeEOL);
  for (auto& [layer, check_list] : check_map) {
    // get rule step map
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kEdge);
    auto& rule_eol_map = rule_map->get_rule_map(RuleType::kEdgeEOL);

    // handle eol edges
    for (auto& point_pair : check_list->get_points()) {
      auto* point_1_next = point_pair.first->nextEndpoint();
      auto* point_2_next = point_pair.second->nextEndpoint();

      // skip edge without two endpoints
      if (point_1_next != point_pair.second && point_2_next != point_pair.first) {
        continue;
      } else if (point_2_next == point_pair.first) {
        std::swap(point_pair.first, point_pair.second);
      }

      // match rule EOL spacing
      if (!checkSpacingEOL(point_pair.first, point_pair.second, layer, rule_eol_map)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionEOL::checkSpacingEOL(DrcBasicPoint* point_prev, DrcBasicPoint* point_next, idb::IdbLayer* layer,
                                          std::map<int, idrc::ConditionRule*> rule_eol_map)
{
  bool b_result = true;

  bool is_vertical = point_prev->get_x() == point_next->get_x();
  DrcDirection spacing_direction = DrcUtil::outsidePolygonDirection(point_prev, point_next);

  int step_edge_length = point_prev->distance(point_next);
  // find rule and check
  for (auto& [value, rule_eol] : rule_eol_map) {
    if (value <= step_edge_length) {
      continue;
    }

    // violation rect data
    int llx = std::numeric_limits<int>::max();
    int lly = std::numeric_limits<int>::max();
    int urx = std::numeric_limits<int>::min();
    int ury = std::numeric_limits<int>::min();
    std::set<int> net_ids;
    net_ids.insert(point_prev->get_id());
    net_ids.insert(point_next->get_id());

    // get rule data
    auto* condition_rule_eol = static_cast<ConditionRuleEOL*>(rule_eol);
    int eol_spacing = condition_rule_eol->get_eol()->get_eol_space();
    int eol_within
        = condition_rule_eol->get_eol()->get_eol_within().has_value() ? condition_rule_eol->get_eol()->get_eol_within().value() : 0;

    // check eol spacing
    bool is_violation = false;
    auto* iter_point = point_prev;
    while (iter_point) {
      if (iter_point->is_eol_spacing_checked()) {
        break;
      }

      iter_point->set_checked_eol_spacing();
      auto* check_neighbour = iter_point->get_neighbour(spacing_direction);
      if (check_neighbour && check_neighbour->is_spacing() && iter_point->distance(check_neighbour->get_point()) < eol_spacing) {
        check_neighbour->get_point()->set_checked_eol_spacing();

        llx = std::min(llx, iter_point->get_x());
        lly = std::min(lly, iter_point->get_y());
        urx = std::max(urx, iter_point->get_x());
        ury = std::max(ury, iter_point->get_y());

        llx = std::min(llx, check_neighbour->get_point()->get_x());
        lly = std::min(lly, check_neighbour->get_point()->get_y());
        urx = std::max(urx, check_neighbour->get_point()->get_x());
        ury = std::max(ury, check_neighbour->get_point()->get_y());

        net_ids.insert(iter_point->get_id());
        net_ids.insert(check_neighbour->get_point()->get_id());

        is_violation = true;
      } else {
        break;
      }

      if (iter_point == point_next) {
        break;
      }

      iter_point = iter_point->get_next();
    }

    // TODO: check eol within

    if (is_violation) {
      b_result = false;

#if 1
      auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
      auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
      auto gtl_pts_2 = DrcUtil::getPolygonPoints(point_prev->get_neighbour(spacing_direction)->get_point());
      auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif

      // create violation
      DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
      auto violation_type = ViolationEnumType::kViolationMinStep;
      auto* violation_manager = _condition_manager->get_violation_manager();
      auto& violation_list = violation_manager->get_violation_list(violation_type);
      violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
    }
  }

  return b_result;
}

}  // namespace idrc