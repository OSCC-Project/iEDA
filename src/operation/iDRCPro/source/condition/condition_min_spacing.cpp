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

#include "condition_manager.h"
#include "engine_layout.h"
#include "idm.h"

namespace idrc {

void DrcConditionManager::checkMinSpacing(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kDefaultSpacing) == _check_select.end()) {
    return;
  }
  /// get min spacing for this layer
  int min_spacing = DrcTechRuleInst->getMinSpacing(layer);
  if (min_spacing <= 0) {
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
      ieda_solver::bloat(rect, direction, expand_length);
      return true;
    } else if (length > min_spacing) {
      /// means wire length
      ieda_solver::shrink(rect, direction, half_min_spacing);
      return false;
    } else {
      /// half_min_spacing < length <= min_spacing
      int shrink_length = std::abs(half_min_spacing - length);
      ieda_solver::shrink(rect, direction, shrink_length);
      return false;
    }
  };

  auto get_violation_rects = [&](std::vector<ieda_solver::GeometryRect>& results) -> std::vector<bool> {
    std::vector<bool> mark_save(results.size(), true);  /// mark violation need to be saved

    for (int i = 0; i < (int) results.size(); i++) {
      auto state_h = get_new_interval(ieda_solver::HORIZONTAL, results[i]);
      auto state_v = get_new_interval(ieda_solver::VERTICAL, results[i]);
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

  /// check polygon self
  {
    auto& origin_polygons = layout->get_layout_engine()->getLayoutPolygons();  /// copy polyset
    for (auto& origin_polygon : origin_polygons) {
      ieda_solver::GeometryPolygonSet origin_polyset;
      origin_polyset += origin_polygon;
      origin_polyset.clean();  /// eliminate overlaps

      for (auto direction : {ieda_solver::HORIZONTAL, ieda_solver::VERTICAL}) {
        auto polyset_copy = origin_polyset;
        std::vector<ieda_solver::GeometryRect> results;

        // ieda_solver::growAnd(polyset, half_min_spacing);

        ieda_solver::bloat(polyset_copy, direction, half_min_spacing);
        polyset_copy.clean();
        ieda_solver::shrink(polyset_copy, direction, half_min_spacing);
        polyset_copy.clean();
        polyset_copy -= origin_polyset;
        polyset_copy.clean();
        polyset_copy.get(results);

        /// save violation
        for (int i = 0; i < (int) results.size(); i++) {
          if (((ieda_solver::upRightX(results[i]) - ieda_solver::lowLeftX(results[i])) < min_spacing
               && direction == ieda_solver::HORIZONTAL)
              || ((ieda_solver::upRightY(results[i]) - ieda_solver::lowLeftY(results[i])) < min_spacing
                  && direction == ieda_solver::VERTICAL)) {
            addViolation(results[i], layer, ViolationEnumType::kDefaultSpacing);
            violation_num++;
          }
        }
      }
#if 0
      // check diagnal spacing >= min spacing
      auto violation_set = origin_polyset;  /// copy polyset
      violation_set.clean();                /// eliminate overlaps
      ieda_solver::growAnd(violation_set, min_spacing / 2);
      violation_set = violation_set - origin_polyset;
      std::vector<ieda_solver::GeometryRect> and_results;
      violation_set.get(and_results);
      std::vector<ieda_solver::GeometryRect> results;
      for (auto rect : and_results) {
        int length = ieda_solver::getWireWidth(rect, ieda_solver::HORIZONTAL);
        int width = ieda_solver::getWireWidth(rect, ieda_solver::VERTICAL);
        if (length < min_spacing && width < min_spacing) {
          ieda_solver::bloat(rect, ieda_solver::HORIZONTAL, min_spacing - length);
          ieda_solver::bloat(rect, ieda_solver::VERTICAL, min_spacing - width);
          results.push_back(rect);
        }
      }

      /// save violation
      for (int i = 0; i < (int) results.size(); i++) {
        addViolation(results[i], layer, ViolationEnumType::kDefaultSpacing);
        violation_num++;
      }
#endif
    }
  }

  /// check different poly
  {
    auto violation_position_set = layout->get_layout_engine()->copyPolyset();  /// copy polyset
    violation_position_set.clean();                                            /// eliminate overlaps
    /// get min spacing for horizontal and vertical spacing < min spacing
    std::vector<ieda_solver::GeometryRect> results;
    ieda_solver::growAnd(violation_position_set, half_min_spacing);
    violation_position_set.get(results);

    std::vector<bool> mark_save = get_violation_rects(results);
    /// save violation
    for (int i = 0; i < (int) results.size(); i++) {
      if (true == mark_save[i]) {
        addViolation(results[i], layer, ViolationEnumType::kDefaultSpacing);
        violation_num++;
      }
    }
#if 0
    /// check diagnal spacing >= min spacing
    auto violation_set = layout->get_layout_engine()->copyPolyset();  /// copy polyset
    violation_set.clean();                                            /// eliminate overlaps
    /// get min spacing for horizontal and vertical spacing < min spacing
    ieda_solver::growAnd(violation_set, min_spacing);

    std::vector<ieda_solver::GeometryRect> and_results;
    violation_set.get(and_results);
    results.clear();
    for (auto rect : and_results) {
      int length = ieda_solver::getWireWidth(rect, ieda_solver::HORIZONTAL);
      int width = ieda_solver::getWireWidth(rect, ieda_solver::VERTICAL);
      if (length < min_spacing && width < min_spacing) {
        ieda_solver::bloat(rect, ieda_solver::HORIZONTAL, min_spacing - length);
        ieda_solver::bloat(rect, ieda_solver::VERTICAL, min_spacing - width);
        results.push_back(rect);
      }
    }
    /// save violation
    for (int i = 0; i < (int) results.size(); i++) {
      addViolation(results[i], layer, ViolationEnumType::kDefaultSpacing);
      violation_num++;
    }
#endif
  }
  // DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Spacing:\t") << violation_num << "\tresults = " << results.size()
  //                                              << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc