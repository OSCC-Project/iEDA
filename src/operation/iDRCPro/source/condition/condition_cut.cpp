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

// #include <boost/polygon/polygon_90_set_concept.hpp>
#include <cstdio>
#include <string>

#include "boost_definition.h"
#include "condition_manager.h"
#include "engine_geometry_creator.h"
#include "engine_layout.h"
#include "idm.h"

namespace idrc {

void DrcConditionManager::checkCutSpacing(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kCutSpacing) == _check_select.end()) {
    printf("Error: not cut spacing!!!");
    return;
  }
  /// get min spacing for this layer
  int min_spacing = DrcTechRuleInst->getCutSpacing(layer);
  if (min_spacing <= 0) {
    printf("Error: min cut spacing is negative!!!");
    return;
  }
  int half_min_spacing = min_spacing / 2;

  /**return
   * true : bloat
   * false : shrink
   */
  auto get_new_interval = [&](ieda_solver::GeometryOrientation direction, ieda_solver::GeometryRect& rect) -> bool {
    int length = ieda_solver::getWireWidth(rect, direction);
    if (length <= half_min_spacing) {
      int expand_length = std::abs(half_min_spacing - length);
      ieda_solver::BLOAT(rect, direction, expand_length);
      return true;
    } else if (length > min_spacing) {
      /// means wire length
      ieda_solver::SHRINK(rect, direction, half_min_spacing);
      return false;
    } else {
      /// half_min_spacing < length <= min_spacing
      int shrink_length = std::abs(half_min_spacing - length);
      ieda_solver::SHRINK(rect, direction, shrink_length);
      return false;
    }
  };

  auto get_violation_rects = [&](std::vector<ieda_solver::GeometryRect>& results) -> std::vector<bool> {
    std::vector<bool> mark_save(results.size(), true);  /// mark violation need to be saved

    for (int i = 0; i < (int) results.size(); i++) {
      auto state_h = get_new_interval(ieda_solver::K_HORIZONTAL, results[i]);
      auto state_v = get_new_interval(ieda_solver::K_VERTICAL, results[i]);
      /// if state_h and state_v are all bloat, result is a diagnal rect,
      /// if rect is a diagnal rect, check diagnal spacing >= min spacing is ok
      if (state_h && state_v) {
        if (ptEuclideanDistance(lowLeftX(results[i]), lowLeftY(results[i]), upRightX(results[i]), upRightY(results[i])) >= min_spacing) {
          mark_save[i] = false;  /// mark as don't save
        }
      }
    }

    return mark_save;
  };

  ieda::Stats states;
  int violation_num = 0;

  /// check different poly
  // without corner to corner
  auto violation_position_set = layout->get_layout_engine()->copyPolyset();  /// copy polyset
  std::vector<ieda_solver::GeometryRect> results;
  ieda_solver::growAnd(violation_position_set, half_min_spacing);
  violation_position_set.get(results);

  std::vector<bool> mark_save = get_violation_rects(results);
  /// save violation
  for (int i = 0; i < (int) results.size(); i++) {
    if (true == mark_save[i]) {
      addViolation(results[i], layer, ViolationEnumType::kCutSpacing);
      violation_num++;
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Cut Spacing:\t") << violation_num << "\tresults = " << results.size());
  //
}

void DrcConditionManager::checkCutArraySpacing(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkCutEnclosure(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkCutOverlap(std::string layer, DrcEngineLayout* layout)
{
  auto shrink_rect = [](ieda_solver::GeometryRect& rect, int value) -> bool {
    ieda_solver::GeometryRect result;
    int with = ieda_solver::getWireWidth(rect, ieda_solver::K_HORIZONTAL);
    int height = ieda_solver::getWireWidth(rect, ieda_solver::K_HORIZONTAL);
    if (with < 2 * value || height < 2 * value) {
      return false;
    }

    ieda_solver::SHRINK(rect, ieda_solver::K_HORIZONTAL, value);
    ieda_solver::SHRINK(rect, ieda_solver::K_VERTICAL, value);

    return true;
  };

  ieda::Stats states;

  ieda_solver::GeometryPolygonSet total_polyset;

  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    auto& sub_polyset_overlap = sub_layout->get_engine()->get_polyset_overlap();

    total_polyset += sub_polyset_overlap;
  }

  int total_drc = 0;
  ieda_solver::gtl::self_intersect(total_polyset);
  std::vector<ieda_solver::GeometryRect> overlaps;
  total_polyset.get(overlaps);

  for (auto& overlap : overlaps) {
    std::vector<ieda_solver::GeometryRect> results;
    ieda_solver::getDefaultRectangles(results, overlap);

    for (auto rect : results) {
      addViolation(rect, layer, ViolationEnumType::kCutShort);
      total_drc++;
    }
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Cut Short:\t") << total_drc << "\tlayer " << layer << "\tnets = " << layout->get_sub_layouts().size()
                                             << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());
}

void DrcConditionManager::checkCutWidth(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkLEF58CutSpacingTable(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kCutSpacing) == _check_select.end()) {
    return;
  }

  auto lef58_cut_spacing_table = DrcTechRuleInst->getLef58CutSpacing(layer);
  if (lef58_cut_spacing_table.empty()) {
    return;
  }

  /// get min spacing for this layer

  auto spacing_table = lef58_cut_spacing_table[0];
  const auto& cutclass = spacing_table->get_cutclass();
  int32_t min_spacing = 0; 

  if (cutclass.get_class_name2_list().empty() || cutclass.get_class_name1_list().empty()) {
    min_spacing = 0;
  } else {
    min_spacing = cutclass.get_cut_spacing(cutclass.get_class_name2_list().size() - 1, cutclass.get_class_name1_list().size() - 1)
                              .get_cut_spacing2()
                              .value_or(0);
  }

  if (min_spacing == 0) {
    return;
  }
  int32_t half_min_spacing = min_spacing / 2;

  /**return
   * true : bloat
   * false : shrink
   */
  auto get_new_interval = [&](ieda_solver::GeometryOrientation direction, ieda_solver::GeometryRect& rect) -> bool {
    int length = ieda_solver::getWireWidth(rect, direction);
    if (length <= half_min_spacing) {
      int expand_length = std::abs(half_min_spacing - length);
      ieda_solver::BLOAT(rect, direction, expand_length);
      return true;
    } else if (length > min_spacing) {
      /// means wire length
      ieda_solver::SHRINK(rect, direction, half_min_spacing);
      return false;
    } else {
      /// half_min_spacing < length <= min_spacing
      int shrink_length = std::abs(half_min_spacing - length);
      ieda_solver::SHRINK(rect, direction, shrink_length);
      return false;
    }
  };

  auto get_violation_rects = [&](std::vector<ieda_solver::GeometryRect>& results) -> std::vector<bool> {
    std::vector<bool> mark_save(results.size(), true);  /// mark violation need to be saved

    for (int i = 0; i < (int) results.size(); i++) {
      auto state_h = get_new_interval(ieda_solver::K_HORIZONTAL, results[i]);
      auto state_v = get_new_interval(ieda_solver::K_VERTICAL, results[i]);
      /// if state_h and state_v are all bloat, result is a diagnal rect,
      /// if rect is a diagnal rect, check diagnal spacing >= min spacing is ok
      if (state_h && state_v) {
        if (ptEuclideanDistance(lowLeftX(results[i]), lowLeftY(results[i]), upRightX(results[i]), upRightY(results[i])) >= min_spacing) {
          mark_save[i] = false;  /// mark as don't save
        }
      }
    }

    return mark_save;
  };

  ieda::Stats states;
  int violation_num = 0;

  /// check different poly
  // without corner to corner
  auto violation_position_set = layout->get_layout_engine()->copyPolyset();  /// copy polyset
  std::vector<ieda_solver::GeometryRect> results;
  ieda_solver::growAnd(violation_position_set, half_min_spacing);
  violation_position_set.get(results);

  std::vector<bool> mark_save = get_violation_rects(results);
  /// save violation
  for (int i = 0; i < (int) results.size(); i++) {
    if (true == mark_save[i]) {
      addViolation(results[i], layer, ViolationEnumType::kCutSpacing);
      violation_num++;
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Cut Spacing:\t") << violation_num << "\tresults = " << results.size());
  
}
}  // namespace idrc