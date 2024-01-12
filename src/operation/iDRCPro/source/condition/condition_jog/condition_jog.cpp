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

#include "condition_jog.h"

#include "condition.h"
#include "idrc_util.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_condition_spacing.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionJog::checkFastMode()
{
  bool b_result = true;

  b_result &= checkSpacingJog();

  return b_result;
}

bool DrcRuleConditionJog::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkSpacingJog();

  return b_result;
}

bool DrcRuleConditionJog::checkSpacingJog()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kSpacingJogToJog);
  for (auto& [layer, check_list] : check_map) {
    // get rule step map
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    auto& rule_eol_map = rule_map->get_rule_map(RuleType::kSpacingJogToJog);

    // handle jog edges
    for (auto& point_pair : check_list->get_points()) {
      // judge if point_pair is checked, then continue
      if (point_pair.first->is_jog_checked() || point_pair.second->is_jog_checked()) {
        continue;
      }
      // match rule jog spacing
      if (!checkSpacingJogSegment(point_pair.first, point_pair.second, layer, rule_eol_map)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionJog::checkSpacingJogSegment(DrcBasicPoint* point1, DrcBasicPoint* point2, idb::IdbLayer* layer,
                                                 std::map<int, std::vector<ConditionRule*>> rule_jog_map)
{
  bool b_result = true;

  point1->set_checked_jog();
  point2->set_checked_jog();

  auto direction = point1->direction(point2);
  auto spacing_direction = DrcDirection::kNone;
  DrcBasicPoint* point_jog_top = nullptr;     //
  DrcBasicPoint* point_jog_bottom = nullptr;  //
  DrcBasicPoint* point_other_polygon = nullptr;
  int jog_within_value = 0;  //

  if (point2->get_neighbour(direction)->is_spacing()) {
    spacing_direction = direction;
    point_jog_top = point2;
    point_jog_bottom = point1;
    point_other_polygon = point2->get_neighbour(direction)->get_point();
    jog_within_value = point1->distance(point2) + point2->get_neighbour(spacing_direction)->get_point()->distance(point2);
  }

  else if (point1->get_neighbour(DrcUtil::oppositeDirection(direction))->is_spacing()) {
    spacing_direction = DrcUtil::oppositeDirection(direction);  //
    point_jog_top = point1;
    point_jog_bottom = point2;
    point_other_polygon = point1->get_neighbour(spacing_direction)->get_point();
    jog_within_value = point1->distance(point2) + point1->get_neighbour(spacing_direction)->get_point()->distance(point1);  //
  } else {
    return b_result;
  }
  if (point_jog_top->get_id() < 0 && point_other_polygon->get_id() < 0) {
    return b_result;
  }

  // find rule and check
  for (auto& [value, rule_jog_list] : rule_jog_map) {
    if (value <= jog_within_value) {
      continue;
    }

    for (auto& rule_jog : rule_jog_list) {
      // violation rect data
      int llx = std::min(point_jog_bottom->get_x(), point_other_polygon->get_x());
      int lly = std::min(point_jog_bottom->get_y(), point_other_polygon->get_y());
      int urx = std::max(point_jog_bottom->get_x(), point_other_polygon->get_x());
      int ury = std::max(point_jog_bottom->get_y(), point_other_polygon->get_y());
      std::set<int> net_ids;
      net_ids.insert(point_jog_bottom->get_id());
      net_ids.insert(point_other_polygon->get_id());
      // get rule data
      auto* condition_rule_jog = static_cast<ConditionRuleJogToJog*>(rule_jog);
      int jog_rule_within = value;
      int jog_rule_prl = 0;
      int jog_rule_width = 0;
      int jog_rule_long_spacing = 0;
      int jog_rule_short_spacing = condition_rule_jog->get_jog_to_jog()->get_short_jog_spacing();
      int jog_rule_jog_width = condition_rule_jog->get_jog_to_jog()->get_jog_width();
      int jog_rule_jog_to_jog_spacing = condition_rule_jog->get_jog_to_jog()->get_jog_to_jog_spacing();
      auto width_list = condition_rule_jog->get_jog_to_jog()->get_width_list();
      for (auto width : width_list) {
        if (width.get_par_within() == jog_rule_within) {
          jog_rule_prl = width.get_par_length();
          jog_rule_width = width.get_width();
          jog_rule_long_spacing = width.get_long_jog_spacing();
        }
      }

      bool is_violation = false;
      int jog_prl = 0;
      int jog_width = 0;
      // int jog_within = 0;
      int jog_to_jog_spacing = 0;
      int jog_jog_width = 0;

      auto walk_to_cliff = [&](DrcBasicPoint* point, DrcDirection direction, DrcDirection avoid_direction,
                               std::function<DrcBasicPoint*(DrcBasicPoint*)> iter_func) {
        auto* point_current = point;
        auto* point_current_neighbour = point_current->get_neighbour(direction);
        auto* point_next = iter_func(point_current);
        auto* point_next_neighbour = point_next->get_neighbour(direction);

        auto* prev_endpoint = point_current->is_endpoint() ? point_current : nullptr;
        DrcCornerType prev_corner_type = prev_endpoint ? prev_endpoint->getCornerType() : DrcCornerType::kNone;

        while (point_current->direction(point_next) != avoid_direction) {
          if (!point_next_neighbour) {
            if (!point_next->is_endpoint()) {
              point_current = point_next;
              point_current_neighbour = point_next_neighbour;
              point_next = iter_func(point_current);
              point_next_neighbour = point_next->get_neighbour(direction);
              continue;
            } else {
              break;
            }
          }

          if (point_next_neighbour->is_edge()) {
            if (!point_next_neighbour->get_point()->get_neighbour(direction)) {
              break;
            }
            // get within
            int jog_within
                = point_next->distance(point_next_neighbour->get_point())
                  + point_next_neighbour->get_point()->get_neighbour(direction)->get_point()->distance(point_next_neighbour->get_point());
            if (jog_rule_within < jog_within) {
              break;
            }
          }
          if (point_next_neighbour->is_spacing()) {
            // get within
            int jog_within = point_next_neighbour->get_point()->distance(point_next);
            if (jog_rule_within < jog_within || (point_next_neighbour->get_point()->get_id() != point_other_polygon->get_id())) {
              break;
            }
          }

          if (point_next->is_endpoint()) {
            auto corner_type = point_next->getCornerType();
            if (corner_type == DrcCornerType::kConcave && prev_corner_type == DrcCornerType::kConcave) {
              // TODO: net width filter
              if (point_next->distance(point_next->get_neighbour(DrcUtil::oppositeDirection(direction))->get_point())) {
                jog_width = point_next->distance(point_next->get_neighbour(DrcUtil::oppositeDirection(direction))->get_point());
              }
              // jog to jog spacing
              jog_to_jog_spacing = point_next->distance(prev_endpoint);

            } else if (corner_type == DrcCornerType::kConvex && prev_corner_type == DrcCornerType::kConvex) {
              if (point_next_neighbour->is_edge()) {
                break;
              }
              //  jog jog width
              jog_jog_width = point_next->distance(prev_endpoint);
            }
            prev_endpoint = point_next;
            prev_corner_type = corner_type;
          }

          // refresh prl
          bool is_direction_vertical = direction == DrcDirection::kUp || direction == DrcDirection::kDown;
          bool is_point_vertical = point_current->get_x() == point_next->get_x();
          if (is_direction_vertical != is_point_vertical) {
            jog_prl += point_current->distance(point_next);
          }

          // refresh violation rect data
          llx = std::min(llx, point_next->get_x());
          lly = std::min(lly, point_next->get_y());
          urx = std::max(urx, point_next->get_x());
          ury = std::max(ury, point_next->get_y());
          net_ids.insert(point_next->get_id());

          // llx = std::min(llx, point_current_neighbour->get_point()->get_x());
          // lly = std::min(lly, point_current_neighbour->get_point()->get_y());
          // urx = std::max(urx, point_current_neighbour->get_point()->get_x());
          // ury = std::max(ury, point_current_neighbour->get_point()->get_y());
          // net_ids.insert(point_current_neighbour->get_point()->get_id());

          if (jog_width > jog_rule_width && jog_prl > jog_rule_prl && jog_jog_width <= jog_rule_jog_width) {
            // calculate jog spacing
            if (jog_rule_short_spacing > point_next->get_neighbour(direction)->get_point()->distance(point_next)
                || jog_to_jog_spacing < jog_rule_jog_to_jog_spacing) {
              is_violation = true;
              break;
            } else if (jog_width > jog_rule_width && jog_prl > jog_rule_prl && jog_jog_width > jog_rule_jog_width) {
              // calculate jog spacing
              if (jog_rule_long_spacing > point_next->get_neighbour(direction)->get_point()->distance(point_next)
                  || jog_to_jog_spacing < jog_rule_jog_to_jog_spacing) {
                is_violation = true;
                break;
              }
            }
          }

          point_current = point_next;
          point_current_neighbour = point_next_neighbour;
          point_next = iter_func(point_current);
          point_next_neighbour = point_next->get_neighbour(direction);
        }
      };

#ifdef DEBUG_IDRC_CONDITION_JOG
      auto gtl_pts_1 = DrcUtil::getPolygonPoints(point1);
      auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
      auto gtl_pts_2 = DrcUtil::getPolygonPoints(point_other_polygon);
      auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif

      auto [avoid_dir_1, avoid_dir_2] = DrcUtil::orthogonalDirections(spacing_direction);
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_prev(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_prev(); });

      auto spacing_direction_opposite = DrcUtil::oppositeDirection(spacing_direction);
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_prev(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_prev(); });
#ifdef DEBUG_IDRC_CONDITION_JOG
      auto vio_rect = DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
#endif
      if (is_violation) {
        // create violation
        DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
        auto violation_type = ViolationEnumType::kViolationJogToJog;
        auto* violation_manager = _condition_manager->get_violation_manager();
        auto& violation_list = violation_manager->get_violation_list(violation_type);
        violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
        break;
      }
    }
  }

  return b_result;
}

}  // namespace idrc