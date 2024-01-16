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
      if (is_same_metal) {
        continue;
      }
      std::set<int> par_metal_id;

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
        auto edges_1 = scanline_dm->getEdgesInRect(query_1[0], query_1[1], query_1[2], query_1[3]);
        auto edges_2 = scanline_dm->getEdgesInRect(query_2[0], query_2[1], query_2[2], query_2[3]);

        auto has_par_edge = [&](DrcBasicPoint* point, std::vector<std::pair<DrcBasicPoint*, DrcBasicPoint*>>& edges) {
          int result = false;
          for (auto& edge : edges) {
            if (edge.first == point || edge.second == point) {
              continue;
            }
            bool is_edge_vertical = edge.first->get_x() == edge.second->get_x();
            if (is_edge_vertical == is_vertical) {
              continue;
            }
            // TODO: same metal 贯穿时需豁免
            par_metal_id.insert(edge.first->get_id());
            result = true;
          }
          return result;
        };
        bool is_par_edge_1 = has_par_edge(point_prev, edges_1);
        bool is_par_edge_2 = has_par_edge(point_next, edges_2);
        if (is_two_edges) {
          if (is_par_edge_1 && is_par_edge_2) {
            is_on = true;
          }
        } else {
          if (is_par_edge_1 || is_par_edge_2) {
            is_on = true;
          }
        }

        if (!is_on) {
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

      auto [neighbour_x, neighbour_y]
          = DrcUtil::transformPoint(point_prev->get_x(), point_prev->get_y(), spacing_direction, eol_spacing - 1);
      auto [corner_1_x, corner_1_y]
          = DrcUtil::transformPoint(neighbour_x, neighbour_y, DrcUtil::oppositeDirection(eol_direction), eol_within);
      auto [corner_2_x, corner_2_y] = DrcUtil::transformPoint(point_next->get_x(), point_next->get_y(), eol_direction, eol_within);

      int llx_query = std::min(corner_1_x, corner_2_x);
      int lly_query = std::min(corner_1_y, corner_2_y);
      int urx_query = std::max(corner_1_x, corner_2_x);
      int ury_query = std::max(corner_1_y, corner_2_y);

      std::set<int> net_ids{point_prev->get_id()};

#ifdef DEBUG_IDRC_CONDITION_EOL
      DrcBasicPoint* point_other = nullptr;
#endif

      auto edges = scanline_dm->getEdgesInRect(llx_query, lly_query, urx_query, ury_query);
      bool is_violation = false;
      for (auto& edge : edges) {
        if (edge.first == point_prev || edge.first == point_next || edge.second == point_prev || edge.second == point_next) {
          continue;
        }
        is_violation = true;
        net_ids.insert(edge.first->get_id());
        if (is_same_metal && par_metal_id.find(edge.first->get_id()) == par_metal_id.end()) {
          is_violation = false;
        }

#ifdef DEBUG_IDRC_CONDITION_EOL
        point_other = edge.first;
#endif

        break;
      }

      if (is_violation) {
#ifdef DEBUG_IDRC_CONDITION_EOL
        auto gtl_pts_1 = DrcUtil::getPolygonPoints(point_prev);
        auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
        auto gtl_pts_2 = point_other ? DrcUtil::getPolygonPoints(point_other) : std::vector<ieda_solver::GtlPoint>();
        auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
        ieda_solver::BgRect rect_query(ieda_solver::BgPoint(llx_query, lly_query), ieda_solver::BgPoint(urx_query, ury_query));
#endif
        // create violation
        DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx_query, lly_query, urx_query, ury_query);
        auto violation_type = ViolationEnumType::kViolationEOL;
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