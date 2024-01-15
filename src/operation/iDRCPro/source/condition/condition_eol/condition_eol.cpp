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
      if (point_pair.first->is_eol_spacing_checked() || point_pair.second->is_eol_spacing_checked()) {
        continue;
      }
      if (point_pair.first->get_id() < 0 && point_pair.second->get_id() < 0) {
        continue;
      }

      auto* point_1_next = point_pair.first->nextEndpoint();
      auto* point_2_next = point_pair.second->nextEndpoint();

      // skip edge without two endpoints
      if (point_1_next != point_pair.second && point_2_next != point_pair.first) {
        continue;
      } else if (point_2_next == point_pair.first) {
        std::swap(point_pair.first, point_pair.second);
      }

      // match rule EOL spacing
      if (!checkSpacingEOLSegment(point_pair.first, point_pair.second, layer, rule_eol_map)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionEOL::checkSpacingEOLSegment(DrcBasicPoint* point_prev, DrcBasicPoint* point_next, idb::IdbLayer* layer,
                                                 std::map<int, std::vector<ConditionRule*>> rule_eol_map)
{
  bool b_result = true;

  auto spacing_direction = DrcUtil::outsidePolygonDirection(point_prev, point_next);
  auto eol_direction = point_prev->direction(point_next);
  bool is_vertical = point_prev->get_x() == point_next->get_x();

  int eol_edge_length = point_prev->distance(point_next);
  // find rule and check
  for (auto& [value, rule_eol_list] : rule_eol_map) {
    if (value <= eol_edge_length) {
      continue;
    }

    for (auto& rule_eol : rule_eol_list) {
      auto* scanline_dm = _engine->get_engine_manager()->get_engine_scanline(layer)->get_data_manager();

      // get rule data
      auto* condition_rule_eol = static_cast<ConditionRuleEOL*>(rule_eol);
      int eol_spacing = condition_rule_eol->get_eol()->get_eol_space();
      int eol_within
          = condition_rule_eol->get_eol()->get_eol_within().has_value() ? condition_rule_eol->get_eol()->get_eol_within().value() : 0;
      bool is_two_edges = condition_rule_eol->get_eol()->get_adj_edge_length().has_value()
                              ? condition_rule_eol->get_eol()->get_adj_edge_length().value().is_two_sides()
                              : false;
      bool has_min_length = condition_rule_eol->get_eol()->get_adj_edge_length().has_value();
      bool parallel_edge = condition_rule_eol->get_eol()->get_parallel_edge().has_value();
      bool is_same_metal = parallel_edge ? condition_rule_eol->get_eol()->get_parallel_edge().value().is_same_metal() : false;
      // TODO: is same metal: 当出现一条边贯穿旁边区域时，发生豁免。也要检查是否同一金属

      if (parallel_edge) {
        bool is_on = false;
        auto parallel_edge_value = condition_rule_eol->get_eol()->get_parallel_edge().value();
        bool is_subtract_eol_width = parallel_edge_value.is_subtract_eol_width();
        int par_space = parallel_edge_value.get_par_space();
        int par_within = parallel_edge_value.get_par_within();
        if (is_subtract_eol_width) {
          par_space -= eol_edge_length;
        }
        auto create_query_region = [&](DrcBasicPoint* p, DrcDirection direction) {
          auto [neighbour_x, neighbour_y] = DrcUtil::transformPoint(p->get_x(), p->get_y(), direction, par_space);
          auto [corner_1_x, corner_1_y] = DrcUtil::transformPoint(neighbour_x, neighbour_y, spacing_direction, eol_within);
          auto [corner_2_x, corner_2_y]
              = DrcUtil::transformPoint(p->get_x(), p->get_y(), DrcUtil::oppositeDirection(spacing_direction), par_within);
          int llx_query = std::min(corner_1_x, corner_2_x);
          int lly_query = std::min(corner_1_y, corner_2_y);
          int urx_query = std::max(corner_1_x, corner_2_x);
          int ury_query = std::max(corner_1_y, corner_2_y);
          return std::vector<int>{llx_query, lly_query, urx_query, ury_query};
        };
        auto query_1 = create_query_region(point_prev, DrcUtil::oppositeDirection(eol_direction));
        auto query_2 = create_query_region(point_next, eol_direction);
        auto points_1 = scanline_dm->getBasicPointsInRect(query_1[0], query_1[1], query_1[2], query_1[3]);
        auto points_2 = scanline_dm->getBasicPointsInRect(query_2[0], query_2[1], query_2[2], query_2[3]);
        auto eol_coord_same = [&](DrcBasicPoint* p, DrcBasicPoint* cmp_p) {
          return is_vertical ? p->get_x() == cmp_p->get_x() : p->get_y() == cmp_p->get_y();
        };
        std::set<int> net_ids;
        auto check_points = [&](std::vector<DrcBasicPoint*> points, DrcBasicPoint* cmp_p) {
          bool result = false;
          for (auto* p : points) {
            if (!eol_coord_same(p, cmp_p)) {
              result = true;
              net_ids.insert(p->get_id());
            }
          }
          return result;
        };
        bool check_one_side = check_points(points_1, point_prev);
        bool check_other_side = check_points(points_2, point_next);
        if (is_two_edges) {
          if (check_one_side && check_other_side) {
            is_on = true;
          }
        } else {
          if (check_one_side || check_other_side) {
            is_on = true;
          }
        }
        if (!is_on || (is_same_metal && net_ids.size() > 1)) {
          continue;
        }
      }

      if (has_min_length) {
        auto adj_edge_length_value = condition_rule_eol->get_eol()->get_adj_edge_length().value();
        if (adj_edge_length_value.get_min_length().has_value()) {
          int min_length = adj_edge_length_value.get_min_length().value();

          int side_min_len = std::min(point_prev->distance(point_prev->prevEndpoint()), point_next->distance(point_next->nextEndpoint()));

          if (side_min_len < min_length) {
            continue;
          }
        }
      }

      auto [neighbour_x, neighbour_y] = DrcUtil::transformPoint(point_prev->get_x(), point_prev->get_y(), spacing_direction, eol_spacing);
      auto [corner_1_x, corner_1_y]
          = DrcUtil::transformPoint(neighbour_x, neighbour_y, DrcUtil::oppositeDirection(eol_direction), eol_within);
      auto [corner_2_x, corner_2_y] = DrcUtil::transformPoint(point_next->get_x(), point_next->get_y(), eol_direction, eol_within);

      int llx_query = std::min(corner_1_x, corner_2_x);
      int lly_query = std::min(corner_1_y, corner_2_y);
      int urx_query = std::max(corner_1_x, corner_2_x);
      int ury_query = std::max(corner_1_y, corner_2_y);
#ifdef DEBUG_IDRC_CONDITION_EOL
      ieda_solver::BgRect rect_query(ieda_solver::BgPoint(llx_query, lly_query), ieda_solver::BgPoint(urx_query, ury_query));
#endif

      auto points = scanline_dm->getBasicPointsInRect(llx_query, lly_query, urx_query, ury_query);
      if (points.size() > 2) {
        // violation rect data
        int llx = std::numeric_limits<int>::max();
        int lly = std::numeric_limits<int>::max();
        int urx = std::numeric_limits<int>::min();
        int ury = std::numeric_limits<int>::min();
        std::set<int> net_ids;

#ifdef DEBUG_IDRC_CONDITION_EOL
        DrcBasicPoint* point_other = nullptr;
#endif
        for (auto* p : points) {
          llx = std::min(llx, p->get_x());
          lly = std::min(lly, p->get_y());
          urx = std::max(urx, p->get_x());
          ury = std::max(ury, p->get_y());

          net_ids.insert(p->get_id());

          p->set_checked_eol_spacing();

#ifdef DEBUG_IDRC_CONDITION_EOL
          if (p->get_id() != point_prev->get_id()) {
            point_other = p;
          }
#endif
        }

        if (is_same_metal && net_ids.size() > 1) {
          continue;
        }

        if (llx != urx && lly != ury) {
#ifdef DEBUG_IDRC_CONDITION_EOL
          auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
          auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
          auto gtl_pts_2 = point_other ? DrcUtil::getPolygonPoints(point_other) : std::vector<ieda_solver::GtlPoint>();
          auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
#endif
          // create violation
          DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
          auto violation_type = ViolationEnumType::kViolationEOL;
          auto* violation_manager = _condition_manager->get_violation_manager();
          auto& violation_list = violation_manager->get_violation_list(violation_type);
          violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
        }
      }

      //       // check eol spacing
      //       bool is_violation = false;
      //       bool is_begin = false;
      //       auto* iter_point = point_prev;
      //       while (iter_point) {
      //         if (iter_point->is_eol_spacing_checked()) {  // TODO: 多条规则时候可能会漏检
      //           break;
      //         }

      //         iter_point->set_checked_eol_spacing();
      //         auto* check_neighbour = iter_point->get_neighbour(spacing_direction);
      //         if (check_neighbour && check_neighbour->is_spacing() && iter_point->distance(check_neighbour->get_point()) < eol_spacing) {
      //           check_neighbour->get_point()->set_checked_eol_spacing();
      //           is_begin = true;

      //           llx = std::min(llx, iter_point->get_x());
      //           lly = std::min(lly, iter_point->get_y());
      //           urx = std::max(urx, iter_point->get_x());
      //           ury = std::max(ury, iter_point->get_y());

      //           llx = std::min(llx, check_neighbour->get_point()->get_x());
      //           lly = std::min(lly, check_neighbour->get_point()->get_y());
      //           urx = std::max(urx, check_neighbour->get_point()->get_x());
      //           ury = std::max(ury, check_neighbour->get_point()->get_y());

      //           net_ids.insert(iter_point->get_id());
      //           net_ids.insert(check_neighbour->get_point()->get_id());

      //           is_violation = true;
      //         } else if (is_begin && check_neighbour) {
      //           break;
      //         }

      //         if (iter_point == point_next) {
      //           break;
      //         }

      //         iter_point = iter_point->get_next();
      //       }

      //       // TODO: check eol within

      //       if (is_violation) {
      //         b_result = false;

      // #ifdef DEBUG_IDRC_CONDITION_EOL
      //         auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
      //         auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
      //         auto* neighbour_point = point_prev->get_neighbour(spacing_direction) ?
      //         point_prev->get_neighbour(spacing_direction)->get_point()
      //                                                                              :
      //                                                                              point_next->get_neighbour(spacing_direction)->get_point();
      //         auto gtl_pts_2 = DrcUtil::getPolygonPoints(neighbour_point);
      //         auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
      // #endif

      //         // create violation
      //         DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
      //         auto violation_type = ViolationEnumType::kViolationEOL;
      //         auto* violation_manager = _condition_manager->get_violation_manager();
      //         auto& violation_list = violation_manager->get_violation_list(violation_type);
      //         violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
      //       }
    }
  }

  return b_result;
}

}  // namespace idrc