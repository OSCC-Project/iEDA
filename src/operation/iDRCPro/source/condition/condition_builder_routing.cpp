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

#include "condition_builder.h"
#include "drc_basic_point.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
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
  // filterSpacing();
}
/**
 *  filter data by min spacing and max spacing
 */
void DrcConditionBuilder::filterSpacing()
{
  auto* engine = _condition_manager->get_engine();
  auto* engine_manager = engine->get_engine_manager();

  /// get routing layer scanline data map
  auto& scanline_map = engine_manager->get_engine_scanlines(LayoutType::kRouting);
  for (auto& [layer, scanline_engine] : scanline_map) {
    /// get min spacing & max spacing
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    int min_spacing = rule_map->get_min();
    int max_spacing = rule_map->get_max();

    int polygon_number = 0;
    /// filter basic points by min spacing & max spacing
    auto* scanline_dm = scanline_engine->get_data_manager();
    auto& basic_pts = scanline_dm->get_basic_points();
    for (int i = 0; i < (int) basic_pts.size(); ++i) {
      auto& basic_point = basic_pts[i];
      if (basic_point->is_start()) {
        filterSpacingForPolygon(basic_point, layer, min_spacing, max_spacing);
        polygon_number++;
      }
    }

    // std::cout << "polygon_number = " << polygon_number << std::endl;

    auto boost_rects = _condition_manager->get_violation_manager()->get_boost_rects(layer);
    if (boost_rects.size() > 0) {
      auto basic_polygons = scanline_dm->get_boost_polygons();
      int a = 0;
    }
  }

  //   std::cout << "kViolationShort number = "
  //             << _condition_manager->get_violation_manager()->get_violation_list(ViolationEnumType::kViolationShort).size() << std::endl;
  //   std::cout << "kViolationMinSpacing number = "
  //             << _condition_manager->get_violation_manager()->get_violation_list(ViolationEnumType::kViolationMinSpacing).size() <<
  //             std::endl;
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
    int max_step_width = rule_step_conditionrule.empty()
                             ? -1
                             : std::max((--rule_step_conditionrule.end())->first, (--rule_step58_conditionrule.end())->first);

    auto& rule_notch_conditionrule = rule_map_edge->get_rule_map(RuleType::kEdgeNotch);
    int max_notch_width = rule_notch_conditionrule.empty() ? -1 : (--rule_notch_conditionrule.end())->first;

    auto* rule_map_spacing = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    auto& rule_jog_conditionrule = rule_map_spacing->get_rule_map(RuleType::kSpacingJogToJog);
    int max_jog_spacing = rule_jog_conditionrule.empty() ? -1 : (--rule_jog_conditionrule.end())->first;

    auto& rule_prl_conditionrule = rule_map_spacing->get_rule_map(RuleType::kSpacintPRLTable);
    int max_prl_spacing = rule_prl_conditionrule.empty() ? -1 : (--rule_prl_conditionrule.end())->first;

    std::map<RuleType, int> max_value_map{{RuleType::kEdgeEOL, max_eol_width},
                                          {RuleType::kEdgeMinStep, max_step_width},
                                          {RuleType::kEdgeNotch, max_notch_width},
                                          {RuleType::kSpacingJogToJog, max_jog_spacing},
                                          {RuleType::kSpacintPRLTable, max_prl_spacing}};

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

/**
 * filter spacing for a polygon, begin checking from start_point
 */
void DrcConditionBuilder::filterSpacingForPolygon(DrcBasicPoint* start_point, idb::IdbLayer* layer, int min_spacing, int max_spacing)
{
  DrcBasicPoint* iter_point = start_point;
  while (iter_point != nullptr) {
    checkSpacing(iter_point, layer, min_spacing, max_spacing, DrcDirection::kUp);
    checkSpacing(iter_point, layer, min_spacing, max_spacing, DrcDirection::kRight);

    /// turn to next point
    iter_point = iter_point->get_next();

    /// if turn to start point, break loop
    if (iter_point == start_point) {
      break;
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

std::vector<ieda_solver::GtlPoint> DrcConditionBuilder::get_boost_point(DrcBasicPoint* point)
{
  std::vector<ieda_solver::GtlPoint> point_list;
  auto* iter_pt = point;
  while (nullptr != iter_pt) {
    ieda_solver::GtlPoint gtl_pt(iter_pt->get_x(), iter_pt->get_y());
    point_list.emplace_back(gtl_pt);

    iter_pt = iter_pt->get_next();
    if (iter_pt == point) {
      break;
    }
  }

  return point_list;
};

void DrcConditionBuilder::checkSpacing(DrcBasicPoint* point, idb::IdbLayer* layer, int min_spacing, int max_spacing, DrcDirection direction)
{
  auto check_id = [](DrcBasicPoint* point, DrcDirection direction) -> bool {
    auto* neighbour = point->get_neighbour(direction);
    if (neighbour == nullptr) {
      /// no need to check
      return false;
    }

    int point_id = point->get_id();
    int neighbout_id = neighbour->get_point()->get_id();

    if ((point_id == neighbout_id) || (point_id < 0 && neighbout_id < 0)) {
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

  auto* violation_manager = _condition_manager->get_violation_manager();

  /// vertical or horizontal
  bool b_vertical = direction == DrcDirection::kUp ? true : false;

  /// if overlap, save violation as short
  auto* neighbour = point->get_neighbour(direction);
  //   if (neighbour->is_overlap()) {
  //     if (false == point->is_overlap_checked()) {
  // #if DEBUG
  //       auto gtl_pts_1 = get_boost_point(point);
  //       auto gtl_pts_2 = get_boost_point(neighbour->get_point());
  //       auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
  //       auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
  //       /// has violation
  //       auto& violation_list = violation_manager->get_violation_list(ViolationEnumType::kViolationShort);
  //       int list_size = violation_list.size();
  //       std::vector<DrcBasicPoint*> seg{point, neighbour->get_point()};
  // #endif

  //       saveViolationSpacing(point, neighbour->get_point(), layer, b_vertical, -1);

  // #if DEBUG
  //       if (violation_list.size() > list_size) {
  //         auto vio = violation_list.at(violation_list.size() - 1);
  //         auto vio_rect = static_cast<idrc::DrcViolationRect*>(vio);
  //         ieda_solver::GtlRect boost_rect(vio_rect->get_llx(), vio_rect->get_lly(), vio_rect->get_urx(), vio_rect->get_ury());
  //         int a = 0;
  //       }
  // #endif
  //     }
  //     return;
  //   }

  /// if spacing
  if (neighbour->is_spacing()) {
    /// if has been checked, return
    bool b_checked = direction == DrcDirection::kUp ? point->is_vertical_checked() : point->is_horizontal_checked();
    if (true == b_checked) {
      return;
    }

    int spacing = b_vertical ? std::abs(neighbour->get_point()->get_y() - point->get_y())
                             : std::abs(neighbour->get_point()->get_x() - point->get_x());
    if (spacing == 0) {
#if DEBUG
      auto gtl_pts_1 = get_boost_point(point);
      auto gtl_pts_2 = get_boost_point(neighbour->get_point());
      auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
      auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());
      int a = 0;
#endif
      return;
    } else if (spacing > max_spacing) {
      /// no violation
      return;
    } else if (spacing < min_spacing) {
#if DEBUG
      auto gtl_pts_1 = get_boost_point(point);
      auto gtl_pts_2 = get_boost_point(neighbour->get_point());
      auto polygon_1 = ieda_solver::GtlPolygon(gtl_pts_1.begin(), gtl_pts_1.end());
      auto polygon_2 = ieda_solver::GtlPolygon(gtl_pts_2.begin(), gtl_pts_2.end());

      auto& violation_list = violation_manager->get_violation_list(ViolationEnumType::kViolationMinSpacing);
      int list_size = violation_list.size();
      std::vector<DrcBasicPoint*> seg{point, neighbour->get_point()};
#endif

      /// has violation
      saveViolationSpacing(point, neighbour->get_point(), layer, b_vertical, min_spacing);

#if DEBUG
      if (violation_list.size() > list_size) {
        auto vio = violation_list.at(violation_list.size() - 1);
        auto vio_rect = static_cast<idrc::DrcViolationRect*>(vio);
        ieda_solver::GtlRect boost_rect(vio_rect->get_llx(), vio_rect->get_lly(), vio_rect->get_urx(), vio_rect->get_ury());
        int a = 0;
      }
#endif
      return;
    } else {
      /// need to check rule
      // auto* check_list = _condition_manager->get_check_list_routing_layer(layer);
      // check_list->addCheckList(point, neighbour->get_point());
    }

    return;
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
  }
}

/**
 * save violation for a pair of vertical points
 * b_vertical : true : start_point_bottom & start_point_top are vertical direction, false : horizontal direction
 * min_spacing == -1 : violation is overlap, min > 0 : min spacing violation occurs
 */
void DrcConditionBuilder::saveViolationSpacing(DrcBasicPoint* start_point_1, DrcBasicPoint* start_point_2, idb::IdbLayer* layer,
                                               bool b_vertical, int min_spacing)
{
  /// need to save
  auto isChecked = [](DrcBasicPoint* start_point, int min_spacing, bool b_vertical) -> bool {
    /// if check overlap
    if (min_spacing == -1) {
      if (start_point->is_overlap_checked()) {
        return true;
      } else {
        start_point->set_checked_overlap();
        return false;
      }
    } else {
      /// set start_point_bottom vertical has been checked
      if (b_vertical) {
        /// virtical
        if (start_point->is_vertical_checked()) {
          return true;
        } else {
          start_point->set_checked_vertical();
          return false;
        }
      } else {
        /// horizontal
        if (start_point->is_horizontal_checked()) {
          return true;
        } else {
          start_point->set_checked_horizontal();
          return false;
        }
      }
    }
  };

  /// if has been checked, skip this check
  if (isChecked(start_point_1, min_spacing, b_vertical)) {
    return;
  }

  auto* violation_manager = _condition_manager->get_violation_manager();

  int llx = std::min(start_point_1->get_x(), start_point_2->get_x());
  int lly = std::min(start_point_1->get_y(), start_point_2->get_y());
  int urx = std::max(start_point_1->get_x(), start_point_2->get_x());
  int ury = std::max(start_point_1->get_y(), start_point_2->get_y());

  std::set<int> net_ids;
  net_ids.insert(start_point_1->get_id());
  net_ids.insert(start_point_2->get_id());

  auto check_polygon_violation = [&isChecked](int& llx, int& lly, int& urx, int& ury, std::set<int>& net_ids, bool b_vertical,
                                              DrcBasicPoint* start_point_1, DrcBasicPoint* start_point_2, bool b_next,
                                              int min_spacing = -1) {
    auto iterate_direction
        = b_vertical ? (b_next ? DrcDirection::kRight : DrcDirection::kLeft) : (b_next ? DrcDirection::kUp : DrcDirection::kDown);
    auto iterate_function = [=](DrcBasicPoint* point) -> DrcBasicPoint* {
      if (min_spacing == -1) {
        if (point->get_neighbour(iterate_direction)) {
          if (!point->get_neighbour(iterate_direction)->is_overlap()) {
            return nullptr;
          }
          return point->get_neighbour(iterate_direction)->get_point();
        }
        return nullptr;
      } else {
        return b_next ? point->get_next() : point->get_prev();
      }
      return nullptr;
    };
    auto* iter_point = iterate_function(start_point_1);
    while (iter_point && start_point_1->direction(iter_point) == iterate_direction) {
      auto neighbour = b_vertical ? iter_point->get_neighbour(DrcDirection::kUp) : iter_point->get_neighbour(DrcDirection::kRight);
      // while (neighbour && neighbour->get_point()->get_x() == iter_point->get_x()
      //        && neighbour->get_point()->get_y() == iter_point->get_y()) {
      //   neighbour = b_vertical ? neighbour->get_point()->get_neighbour(DrcDirection::kUp)
      //                          : neighbour->get_point()->get_neighbour(DrcDirection::kRight);
      // }
      if (neighbour == nullptr) {
        break;
      }
      if (min_spacing == -1) {
        if (!neighbour->is_overlap()) {
          break;
        }
      } else {
        if (!neighbour->is_spacing()) {
          break;
        }
        if (neighbour->get_point()->get_id() != start_point_2->get_id() || iter_point->distance(neighbour->get_point()) >= min_spacing) {
          break;
        }
      }

      /// is checked
      if (isChecked(iter_point, min_spacing, b_vertical)) {
        break;
      }

      /// build violation rect
      llx = std::min(llx, iter_point->get_x());
      lly = std::min(lly, iter_point->get_y());
      llx = std::min(llx, neighbour->get_point()->get_x());
      lly = std::min(lly, neighbour->get_point()->get_y());

      urx = std::max(urx, iter_point->get_x());
      ury = std::max(ury, iter_point->get_y());
      urx = std::max(urx, neighbour->get_point()->get_x());
      ury = std::max(ury, neighbour->get_point()->get_y());

      net_ids.insert(iter_point->get_id());
      net_ids.insert(neighbour->get_point()->get_id());

      iter_point = iterate_function(iter_point);
    }
  };

  auto check_violation = [&isChecked](int& llx, int& lly, int& urx, int& ury, std::set<int>& net_ids, bool b_vertical,
                                      DrcBasicPoint* start_point_1, DrcBasicPoint* start_point_2, bool b_next, int min_spacing = -1) {
    /// iterate front
    DrcDirection iterate_direction
        = b_vertical ? (b_next ? DrcDirection::kRight : DrcDirection::kLeft) : (b_next ? DrcDirection::kUp : DrcDirection::kDown);
    auto* iter_point = start_point_1->get_neighbour(iterate_direction);
    while (iter_point != nullptr) {
      if (iter_point->is_spacing() || iter_point->get_type().isType(ScanlineDataType::kNone)) {
        break;
      }

      auto neighbour = b_vertical ? iter_point->get_point()->get_neighbour(DrcDirection::kUp)
                                  : iter_point->get_point()->get_neighbour(DrcDirection::kRight);
      if (neighbour == nullptr) {
        break;
      }

      if (min_spacing == -1) {
        /// check overlap
        if (false == neighbour->is_overlap() || iter_point->get_point()->get_id() == neighbour->get_point()->get_id()) {
          break;
        }
      } else {
        /// check min spacing
        if (false == neighbour->is_spacing()) {
          break;
        }
      }

      /// check if same violation
      bool is_same_violation = b_vertical ? neighbour->get_point()->get_y() == start_point_2->get_y()
                                          : neighbour->get_point()->get_x() == start_point_2->get_x();
      if (false == is_same_violation) {
        break;
      }

      /// is checked
      if (isChecked(iter_point->get_point(), min_spacing, b_vertical)) {
        break;
      }

      /// build violation rect
      llx = std::min(llx, iter_point->get_point()->get_x());
      lly = std::min(lly, iter_point->get_point()->get_y());
      llx = std::min(llx, neighbour->get_point()->get_x());
      lly = std::min(lly, neighbour->get_point()->get_y());

      urx = std::max(urx, iter_point->get_point()->get_x());
      ury = std::max(ury, iter_point->get_point()->get_y());
      urx = std::max(urx, neighbour->get_point()->get_x());
      ury = std::max(ury, neighbour->get_point()->get_y());

      net_ids.insert(iter_point->get_point()->get_id());
      net_ids.insert(neighbour->get_point()->get_id());

      /// turn to next front point
      iter_point = iter_point->get_point()->get_neighbour(iterate_direction);
    }
  };

  check_polygon_violation(llx, lly, urx, ury, net_ids, b_vertical, start_point_1, start_point_2, true, min_spacing);
  check_polygon_violation(llx, lly, urx, ury, net_ids, b_vertical, start_point_1, start_point_2, false, min_spacing);

  /// save violation as rect
  if (llx == urx || lly == ury) {
    // skip area 0
    return;
  }
  DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, llx, lly, urx, ury);
  auto violation_type = min_spacing == -1 ? ViolationEnumType::kViolationShort : ViolationEnumType::kViolationMinSpacing;
  auto& violation_list = violation_manager->get_violation_list(violation_type);
  violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
}

void DrcConditionBuilder::buildWidth()
{
  auto* engine = _condition_manager->get_engine();
  auto* engine_manager = engine->get_engine_manager();

  int number = 0;
  int max_number = 0;
  int overlap_number = 0;
  int min_spacing_number = 0;
  std::vector<std::pair<int, int>> vio_overlaps;
  std::vector<std::pair<int, int>> vio_min_spacings;
  //   for (auto& [type, scanline_map] : engine_manager->get_scanline_matrix()) {
  auto& scanline_map = engine_manager->get_engine_scanlines(LayoutType::kRouting);
  for (auto& [layer, scanline_engine] : scanline_map) {
    if (layer->get_id() == 0) {
      continue;
    }
    auto* scanline_dm = scanline_engine->get_data_manager();
    auto& basic_pts = scanline_dm->get_basic_points();
    max_number += basic_pts.size();
#pragma omp parallel for
    for (int i = 0; i < (int) basic_pts.size(); ++i) {
      auto& basic_point = basic_pts[i];

#pragma omp parallel sections
      {
#pragma omp section
        {
          /// only check direction of up and right
          auto* neighbour_up = basic_point->get_neighbour(DrcDirection::kUp);
          if (neighbour_up != nullptr && neighbour_up->is_overlap()) {
#pragma omp critical
            {
              overlap_number++;
              vio_overlaps.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
              vio_overlaps.push_back(std::make_pair<int, int>(neighbour_up->get_point()->get_x(), neighbour_up->get_point()->get_y()));
            }
          }

          if (neighbour_up != nullptr && neighbour_up->is_spacing()) {
            int spacing = neighbour_up->get_point()->get_y() - basic_point->get_y();
            if (spacing < 100 && !neighbour_up->is_overlap()) {
#pragma omp critical
              {
                min_spacing_number++;
                vio_min_spacings.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
                vio_min_spacings.push_back(
                    std::make_pair<int, int>(neighbour_up->get_point()->get_x(), neighbour_up->get_point()->get_y()));
              }
            }
            if (spacing >= 100 && spacing <= 1000) {
#pragma omp critical
              {
                number++;
              }
            }
          }
        }

#pragma omp section
        {
          auto* neighbour_right = basic_point->get_neighbour(DrcDirection::kRight);
          if (neighbour_right != nullptr && neighbour_right->is_overlap()) {
#pragma omp critical
            {
              overlap_number++;
              vio_overlaps.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
              vio_overlaps.push_back(
                  std::make_pair<int, int>(neighbour_right->get_point()->get_x(), neighbour_right->get_point()->get_y()));
            }
          }

          if (neighbour_right != nullptr && neighbour_right->is_spacing()) {
            int spacing = neighbour_right->get_point()->get_x() - basic_point->get_x();
            if (spacing < 100 && !neighbour_right->is_overlap()) {
#pragma omp critical
              {
                min_spacing_number++;
                vio_min_spacings.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
                vio_min_spacings.push_back(
                    std::make_pair<int, int>(neighbour_right->get_point()->get_x(), neighbour_right->get_point()->get_y()));
              }
            }
            if (spacing >= 100 && spacing <= 1000) {
#pragma omp critical
              {
                number++;
              }
            }
          }
        }
      }
    }
  }

  int a = 0;
}

// void DrcConditionBuilder::checkEOL(DrcBasicPoint* point, ScanlineNeighbour* neighbour, bool is_vertical)
// {
//   DrcDirection check_direciton = DrcDirection::kNone;
//   if (is_vertical) {
//     if (point->get_neighbour(DrcDirection::kUp)->is_spacing()) {
//       check_direciton = DrcDirection::kUp;
//     } else if (point->get_neighbour(DrcDirection::kDown)->is_spacing()) {
//       check_direciton = DrcDirection::kDown;
//     }
//   }

//   else if (is_vertical == false) {
//     if (point->get_neighbour(DrcDirection::kUp)->is_spacing()) {
//       // todo
//     } else if (true) {
//       // todo
//     }
//   }

//   auto* point_next = point->get_neighbour(check_direciton)->get_point();
//   auto* neighbour_next = neighbour->get_point()->get_neighbour(check_direciton)->get_point();

//   // int point_length = point_next->distance(point);
//   // auto* point_length = std::abs(point_next->get_y() - point->get_y());
//   // int neighbour_length = std::abs(neighbour_next->get_y() - neighbour->get_point()->get_y());

//   // ------------------------
//   if ((is_vertical == false) && point->get_neighbour(DrcDirection::kUp)->is_spacing()) {
//     auto* point_next = point->get_neighbour(DrcDirection::kUp)->get_point();
//     auto* neighbour_next = neighbour->get_point()->get_neighbour(DrcDirection::kUp)->get_point();

//     int point_length = std::abs(point_next->get_y() - point->get_y());
//     int neighbour_length = std::abs(neighbour_next->get_y() - neighbour->get_y());
//   }

//   if ((is_vertical == false) && point->get_neighbour(DrcDirection::kDown)->is_spacing()) {
//     auto* point_next = point->get_neighbour(DrcDirection::kDown)->get_point();
//     auto* neighbour_next = neighbour->get_point()->get_neighbour(DrcDirection::kDown)->get_point();

//     int point_length = std::abs(point_next->get_y() - point->get_y());
//     int neighbour_length = std::abs(neighbour_next->get_y() - neighbour->get_point()->get_y());
//   }

//   if ((is_vertical == true) || (point->get_neighbour(DrcDirection::kRight)->is_spacing)) {
//     auto* point_next = point->get_neighbour(DrcDirection::kRight)->get_point();
//     auto* neighbour_next = neighbour->get_point()->get_neighbour(DrcDirection::kRight)->get_point();

//     int point_length = std::abs(point_next->get_x() - point->get_x());
//     int neighbour_length = std::abs(neighbour_next->get_x() - neighbour->get_point()->get_x());
//   }

//   if ((is_vertical == true) || (point->get_neighbour(DrcDirection::kLeft)->is_spacing)) {
//     auto* point_next = point->get_neighbour(DrcDirection::kLeft)->get_point();
//     auto* neighbour_next = neighbour->get_point()->get_neighbour(DrcDirection::kLeft)->get_point();

//     int point_length = std::abs(point_next->get_x() - point->get_x());
//     int neighbour_length = std::abs(neighbour_next->get_x() - neighbour->get_point()->get_x());
//   }

//   if (point_length <= max_value_map[RuleType::kEdgeEOL] || neighbour_next <= max_value_map[RuleType::kEdgeEOL]) {
//     // TODO: add edge to bucket

//     auto* check_list = _condition_manager->get_check_list(RuleType::kEdgeEOL, layer);
//     check_list->addCheckList(point, neighbour->get_point());
//     checkEOL(point, neighbour);
//   }
// }

}  // namespace idrc