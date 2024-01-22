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

#include <cmath>

#include "DRCViolationType.h"
#include "condition_builder.h"
#include "drc_basic_point.h"
#include "idrc_violation.h"
#include "idrc_violation_manager.h"
#include "tech_rules.h"

namespace idrc {

#define DEBUG 0

/**
 * build condition for routing layers
 */
void DrcConditionBuilder::buildConditionRoutingLayer()
{
  filterEdge();
}

/// @brief filter shape by Edge and Value
void DrcConditionBuilder::filterEdge()
{
  auto* engine = _condition_manager->get_engine();
  auto* engine_manager = engine->get_engine_manager();

  /// get routing layer scanline data map
  auto& scanline_map = engine_manager->get_engine_scanlines(LayoutType::kRouting);
  for (auto& [layer, scanline_engine] : scanline_map) {
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map_edge = rule_routing_layer->get_condition_map(RuleType::kEdge);
    auto& rule_eol_conditionrule = rule_map_edge->get_rule_map(RuleType::kEdgeEOL);
    int max_eol_width = rule_eol_conditionrule.empty() ? -1 : (--rule_eol_conditionrule.end())->first;
    // avoid null rule ： map first max

    auto& rule_step58_conditionrule = rule_map_edge->get_rule_map(RuleType::kEdgeMinStepLef58);
    auto& rule_step_conditionrule = rule_map_edge->get_rule_map(RuleType::kEdgeMinStep);
    int max_step_normal_width = rule_step_conditionrule.empty() ? -1 : (--rule_step_conditionrule.end())->first;
    int max_step_lef58_width = rule_step58_conditionrule.empty() ? -1 : (--rule_step58_conditionrule.end())->first;
    int max_step_width = std::max(max_step_normal_width, max_step_lef58_width);

    auto& rule_notch_conditionrule = rule_map_edge->get_rule_map(RuleType::kEdgeNotch);
    int max_notch_width = rule_notch_conditionrule.empty() ? -1 : (--rule_notch_conditionrule.end())->first;

    auto* rule_map_spacing = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    auto& rule_jog_conditionrule = rule_map_spacing->get_rule_map(RuleType::kSpacingJogToJog);
    int max_jog_spacing = rule_jog_conditionrule.empty() ? -1 : (--rule_jog_conditionrule.end())->first;

    auto& rule_prl_conditionrule = rule_map_spacing->get_rule_map(RuleType::kSpacintPRLTable);
    int max_prl_spacing = rule_prl_conditionrule.empty() ? -1 : (--rule_prl_conditionrule.end())->first;

    auto* rule_map_area = rule_routing_layer->get_condition_map(RuleType::kArea);
    auto& rule_area_conditionrule = rule_map_area->get_rule_map(RuleType::kAreaMin);
    auto& rule_area_lef58_conditionrule = rule_map_area->get_rule_map(RuleType::kAreaLef58);
    int max_area_exept_edge_length = rule_area_conditionrule.empty() ? -1 : (--rule_area_conditionrule.end())->first;
    int max_area_lef58_exept_edge_length = rule_area_lef58_conditionrule.empty() ? -1 : (--rule_area_lef58_conditionrule.end())->first;
    int max_area_exept_edge = std::max(max_area_exept_edge_length, max_area_lef58_exept_edge_length);

    std::map<RuleType, int> max_value_map{{RuleType::kEdgeEOL, max_eol_width},           {RuleType::kEdgeMinStep, max_step_width},
                                          {RuleType::kEdgeNotch, max_notch_width},       {RuleType::kSpacingJogToJog, max_jog_spacing},
                                          {RuleType::kSpacintPRLTable, max_prl_spacing}, {RuleType::kArea, max_area_exept_edge}};

    auto* scanline_dm = scanline_engine->get_data_manager();
    auto& basic_pts = scanline_dm->get_basic_points();
    for (int i = 0; i < (int) basic_pts.size(); ++i) {
      auto& basic_point = basic_pts[i];
      if (basic_point->is_start()) {
        filterEdgeForPolygon(basic_point, max_value_map, layer);
      }
    }
  }
}

void DrcConditionBuilder::filterEdgeForPolygon(DrcBasicPoint* start_point, std::map<RuleType, int>& max_value_map, idb::IdbLayer* layer)
{
  DrcBasicPoint* iter_point = start_point;
  while (iter_point != nullptr) {
    checkEdge(iter_point, max_value_map, layer, DrcDirection::kUp);
    checkEdge(iter_point, max_value_map, layer, DrcDirection::kRight);

    /// turn to next point
    iter_point = iter_point->get_next();

    /// if turn to start point, break loop
    if (iter_point == start_point) {
      break;
    }
  }
}

void DrcConditionBuilder::checkEdge(DrcBasicPoint* point, std::map<RuleType, int>& max_value_map, idb::IdbLayer* layer,
                                    DrcDirection direction)
{
  auto check_id = [](DrcBasicPoint* point, DrcDirection direction) -> bool {
    auto* neighbour = point->get_neighbour(direction);
    if (neighbour == nullptr) {
      /// no need to check
      return false;
    }

    /// need to check
    return true;
  };

  /// compare id between point and neighbour if need to check
  if (check_id(point, direction) == false) {
    return;
  }

  /// vertical or horizontal
  bool b_vertical = direction == DrcDirection::kUp ? true : false;

  // TODO: 不需要检测的规则也无需分发

  /// if overlap, save violation as short
  auto* neighbour = point->get_neighbour(direction);
  if (neighbour->is_overlap()) {
    auto* check_list = _condition_manager->get_check_list(RuleType::kConnectivity, layer);
    check_list->addCheckList(point, neighbour->get_point());
  } else if (neighbour->is_edge()) {
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      return;
    }

    // get environment type
    auto* neighbour_prev = b_vertical ? point->get_neighbour(DrcDirection::kDown) : point->get_neighbour(DrcDirection::kLeft);
    auto* neighbour_next = b_vertical ? neighbour->get_point()->get_neighbour(DrcDirection::kUp)
                                      : neighbour->get_point()->get_neighbour(DrcDirection::kRight);
    // define none_or_spacing function
    auto neighbour_none_or_spacing = [](ScanlineNeighbour* neighbour) { return !neighbour || neighbour->is_spacing(); };

    int edge_length = b_vertical ? std::abs(neighbour->get_point()->get_y() - point->get_y())
                                 : std::abs(neighbour->get_point()->get_x() - point->get_x());

    if (neighbour_none_or_spacing(neighbour_prev) && neighbour_none_or_spacing(neighbour_next)) {
      // S/E/S or N/E/N or N/E/S or S/E/N : EOL
      if (edge_length <= max_value_map[RuleType::kEdgeEOL]) {
        // add edge to EOL bucket
        auto* check_list = _condition_manager->get_check_list(RuleType::kEdgeEOL, layer);
        check_list->addCheckList(point, neighbour->get_point());
      }
    }
    if ((neighbour_none_or_spacing(neighbour_prev) && neighbour_next && neighbour_next->is_width())
        || (neighbour_none_or_spacing(neighbour_next) && neighbour_prev && neighbour_prev->is_width())) {
      // S/E/W or W/E/S : Jog
      // S/E/W or N/E/W or W/E/S or W/E/N : Step

      if (neighbour_prev && neighbour_next) {
        int within_value = edge_length + neighbour_next->is_spacing() ? neighbour_next->get_point()->distance(neighbour->get_point())
                                                                      : neighbour_prev->get_point()->distance(point);
        if (within_value <= max_value_map[RuleType::kSpacingJogToJog]) {
          // add edge to Jog bucket
          auto* check_list = _condition_manager->get_check_list(RuleType::kSpacingJogToJog, layer);
          check_list->addCheckList(point, neighbour->get_point());
        }
      }

      if (edge_length <= max_value_map[RuleType::kEdgeMinStep]) {
        // add edge to Step bucket
        auto* check_list = _condition_manager->get_check_list(RuleType::kEdgeMinStep, layer);
        check_list->addCheckList(point, neighbour->get_point());
      }
    }
    if ((neighbour_next && neighbour_next->is_width()) && (neighbour_prev && neighbour_prev->is_width())) {
      // W/E/W : Notch
      if (edge_length <= max_value_map[RuleType::kEdgeNotch]) {
        // add edge to Notch bucket
        auto* check_list = _condition_manager->get_check_list(RuleType::kEdgeNotch, layer);
        check_list->addCheckList(point, neighbour->get_point());
      }
    }
    auto filter_prl_spacing = [&](DrcBasicPoint* point, ScanlineNeighbour* neighbour) {
      // S/E : PRL
      if (neighbour && neighbour->is_spacing()) {
        int spacing_value = point->distance(neighbour->get_point());
        if (spacing_value == 0) {
          auto* check_list = _condition_manager->get_check_list(RuleType::kConnectivity, layer);
          check_list->addCheckList(point, neighbour->get_point());
        } else if (point->get_id() != neighbour->get_point()->get_id() && spacing_value <= max_value_map[RuleType::kSpacintPRLTable]) {
          // add edge to PRL bucket
          auto* check_list = _condition_manager->get_check_list(RuleType::kSpacintPRLTable, layer);
          check_list->addCheckList(point, neighbour->get_point());
        }
      }
    };
    filter_prl_spacing(neighbour->get_point(), neighbour_next);
    filter_prl_spacing(point, neighbour_prev);

    if (edge_length >= max_value_map[RuleType::kArea] && point->is_endpoint() && neighbour->get_point()->is_endpoint()
        && point->get_id() == neighbour->get_point()->get_id()) {
      // TODO: use polygon id, not net id
      auto* except_list = _condition_manager->get_except_list(RuleType::kArea, layer);
      except_list->addExceptList(point->get_id());
    }
  }
}

}  // namespace idrc