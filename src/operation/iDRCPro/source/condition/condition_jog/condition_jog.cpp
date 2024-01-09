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
      // TODO: judge if point_pair is checked, then continue

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

  auto direction = point1->direction(point2);
  auto spacing_direction = DrcDirection::kNone;
  DrcBasicPoint* point_jog_top = nullptr;
  DrcBasicPoint* point_jog_bottom = nullptr;
  DrcBasicPoint* point_other_polygon = nullptr;

  if (point2->get_neighbour(direction)->is_spacing()) {
    spacing_direction = direction;
    point_jog_top = point2;
    point_jog_bottom = point1;
    point_other_polygon = point2->get_neighbour(direction)->get_point();
  } else if (point1->get_neighbour(DrcUtil::oppositeDirection(direction))->is_spacing()) {
    spacing_direction = DrcUtil::oppositeDirection(direction);
    point_jog_top = point1;
    point_jog_bottom = point2;
    point_other_polygon = point1->get_neighbour(spacing_direction)->get_point();
  } else {
    return b_result;
  }
  int jog_within_value = point1->distance(point2) + point1->get_neighbour(spacing_direction)->get_point()->distance(point1);

  // find rule and check
  for (auto& [value, rule_jog_list] : rule_jog_map) {
    if (value <= jog_within_value) {
      continue;
    }

    for (auto& rule_jog : rule_jog_list) {
      // TODO: net width filter

      // violation rect data
      int llx = std::numeric_limits<int>::max();
      int lly = std::numeric_limits<int>::max();
      int urx = std::numeric_limits<int>::min();
      int ury = std::numeric_limits<int>::min();
      std::set<int> net_ids;
      net_ids.insert(point1->get_id());
      net_ids.insert(point2->get_id());

      // get rule data
      auto* condition_rule_jog = static_cast<ConditionRuleJogToJog*>(rule_jog);

      bool is_violation = false;
      int prl = 0;

      auto walk_to_cliff = [&](DrcBasicPoint* point, DrcDirection direction, DrcDirection avoid_direction,
                               std::function<DrcBasicPoint*(DrcBasicPoint*)> iter_func) {
        auto* point_current = point;
        auto* point_next = point_current->get_neighbour(direction)->get_point();
        auto* point_next_neighbour = point_next->get_neighbour(direction);

        auto* prev_endpoint = point_current->is_endpoint() ? point_current : nullptr;
        DrcCornerType prev_corner_type = prev_endpoint ? prev_endpoint->getCornerType() : DrcCornerType::kNone;

        while (point_current->direction(point_next) != avoid_direction) {
          // TODO: set point is checked
          // TODO: refresh violation rect data

          if (point_next_neighbour->is_spacing()) {
            // TODO: calculate jog spacing
          } else if (point_next_neighbour->is_edge()) {
            // TODO: get within
          }

          if (point_next->is_endpoint()) {
            auto corner_type = point_next->getCornerType();
            if (corner_type == DrcCornerType::kConcave && prev_corner_type == DrcCornerType::kConcave) {
              // TODO: jog to jog spacing
            } else if (corner_type == DrcCornerType::kConvex && prev_corner_type == DrcCornerType::kConvex) {
              // TODO: jog width
            }

            prev_endpoint = point_next;
            prev_corner_type = corner_type;
          }

          // TODO: refresh prl
          // TODO: use width, spacing, jog to jog spacing to judge if violation region then break

          point_current = point_next;
          point_next = point_current->get_neighbour(direction)->get_point();
          point_next_neighbour = point_next->get_neighbour(direction);
        }
      };

      auto [avoid_dir_1, avoid_dir_2] = DrcUtil::getOrthogonalDirection(spacing_direction);
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_prev(); });
      walk_to_cliff(point_jog_top, spacing_direction, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_prev(); });

      auto spacing_direction_opposite = DrcUtil::oppositeDirection(spacing_direction);
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_next(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_1, [&](DrcBasicPoint* point) { return point->get_prev(); });
      walk_to_cliff(point_other_polygon, spacing_direction_opposite, avoid_dir_2, [&](DrcBasicPoint* point) { return point->get_prev(); });
    }
  }

  return b_result;
}

}  // namespace idrc