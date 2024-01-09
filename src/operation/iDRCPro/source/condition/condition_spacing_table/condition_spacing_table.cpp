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

#include "condition_spacing_table.h"

#include "condition.h"
#include "idrc_util.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_condition_spacing.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionSpacingTable::checkFastMode()
{
  bool b_result = true;

  b_result &= checkSpacingTable();

  return b_result;
}

bool DrcRuleConditionSpacingTable::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkSpacingTable();

  return b_result;
}

bool DrcRuleConditionSpacingTable::checkSpacingTable()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kSpacintPRLTable);
  for (auto& [layer, check_list] : check_map) {
    // get rule step map
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    auto& rule_eol_map = rule_map->get_rule_map(RuleType::kSpacintPRLTable);

    // handle edges aside spacing
    for (auto& point_pair : check_list->get_points()) {
      if (point_pair.first->is_spacing_table_checked() || point_pair.second->is_spacing_table_checked()) {
        continue;
      }

      // match rule spacing table
      if (!checkSpacingTableSegment(point_pair.first, point_pair.second, layer, rule_eol_map)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionSpacingTable::checkSpacingTableSegment(DrcBasicPoint* point, DrcBasicPoint* neighbour, idb::IdbLayer* layer,
                                                            std::map<int, std::vector<ConditionRule*>> rule_spacing_table_map)
{
  bool b_result = true;

  point->set_checked_spacing_table();
  neighbour->set_checked_spacing_table();

  if (rule_spacing_table_map.empty()) {
    return b_result;
  }
  // violation rect data
  int llx = std::min(point->get_x(), neighbour->get_x());
  int lly = std::min(point->get_y(), neighbour->get_y());
  int urx = std::max(point->get_x(), neighbour->get_x());
  int ury = std::max(point->get_y(), neighbour->get_y());
  std::set<int> net_ids;
  net_ids.insert(point->get_id());
  net_ids.insert(neighbour->get_id());

  // calc prl
  auto is_segment_vertical = [](DrcBasicPoint* point1, DrcBasicPoint* point2) { return point1->get_x() == point2->get_x(); };
  auto get_ortho_edge = [&](DrcBasicPoint* p, bool is_vertical) {
    if (p->is_endpoint()) {
      if (is_segment_vertical(p, p->get_prev()) == is_vertical) {
        return std::make_pair(p->prevEndpoint(), p);
      } else {
        return std::make_pair(p, p->nextEndpoint());
      }
    } else {
      return std::make_pair<DrcBasicPoint*, DrcBasicPoint*>(p->prevEndpoint(), p->nextEndpoint());
    }
  };
  auto get_coord = [](DrcBasicPoint* p, bool is_vertical) { return is_vertical ? p->get_y() : p->get_x(); };
  auto get_prl = [&](DrcBasicPoint* point1, DrcBasicPoint* point2) {
    bool is_vertical = is_segment_vertical(point1, point2);
    auto segment1 = get_ortho_edge(point1, !is_vertical);
    auto segment2 = get_ortho_edge(point2, !is_vertical);
    std::vector<DrcBasicPoint*> points{segment1.first, segment1.second, segment2.first, segment2.second};
    std::sort(points.begin(), points.end(), [&](DrcBasicPoint* p1, DrcBasicPoint* p2) {
      int coord1 = get_coord(p1, !is_vertical);
      int coord2 = get_coord(p2, !is_vertical);
      if (coord1 == coord2) {
        return p1->get_id() < p2->get_id();
      }
      return coord1 < coord2;
    });

    for (int i = 1; i < 3; ++i) {
      llx = std::min(llx, points[i]->get_x());
      lly = std::min(lly, points[i]->get_y());
      urx = std::max(urx, points[i]->get_x());
      ury = std::max(ury, points[i]->get_y());

      net_ids.insert(points[i]->get_id());
      points[i]->set_checked_spacing_table();
    }

    return std::abs(get_coord(points[2], !is_vertical) - get_coord(points[1], !is_vertical));
  };
  int prl = get_prl(point, neighbour);

  int spacing_value = point->distance(neighbour);
  int min_spacing = (*rule_spacing_table_map.begin()).first;
  if (spacing_value < min_spacing) {
    b_result = false;
#ifdef DEBUG_IDRC_CONDITION_SPACING_TABLE
    auto gtl_pts_1 = DrcUtil::getPolygonPoints(point);
    auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
    auto gtl_pts_2 = DrcUtil::getPolygonPoints(neighbour);
    auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif
    // create violation
    DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
    auto violation_type = ViolationEnumType::kViolationMinSpacing;
    auto* violation_manager = _condition_manager->get_violation_manager();
    auto& violation_list = violation_manager->get_violation_list(violation_type);
    violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));

    return b_result;
  }

  // TODO: calc width
  int width = 0;

  // find rule and check
  for (auto& [value, rule_prl_list] : rule_spacing_table_map) {
    if (value <= spacing_value) {
      continue;
    }

    for (auto& rule_prl : rule_prl_list) {
      // get rule data
      auto* condition_rule_prl = static_cast<ConditionRuleSpacingPRL*>(rule_prl);

      if (condition_rule_prl->isMatchCondition(width, prl)) {
        b_result = false;
#ifdef DEBUG_IDRC_CONDITION_SPACING_TABLE
        auto gtl_pts_1 = DrcUtil::getPolygonPoints(point);
        auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
        auto gtl_pts_2 = DrcUtil::getPolygonPoints(neighbour);
        auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif
        // create violation
        DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
        auto violation_type = ViolationEnumType::kViolationPRL;
        auto* violation_manager = _condition_manager->get_violation_manager();
        auto& violation_list = violation_manager->get_violation_list(violation_type);
        violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
      }
    }
  }

  return b_result;
}

}  // namespace idrc