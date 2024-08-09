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

  auto get_new_interval = [&](ieda_solver::GeometryOrientation direction, ieda_solver::GeometryRect& rect) {
    int length = ieda_solver::getWireWidth(rect, direction);
    if (length <= half_min_spacing) {
      int expand_length = std::abs(half_min_spacing - length);
      ieda_solver::bloat(rect, direction, expand_length);
    } else if (length > min_spacing) {
      /// means wire length
      ieda_solver::shrink(rect, direction, half_min_spacing);
    } else {
      /// half_min_spacing < length <= min_spacing
      int shrink_length = std::abs(half_min_spacing - length);
      ieda_solver::shrink(rect, direction, shrink_length);
    }
  };

  ieda::Stats states;
  std::vector<ieda_solver::GeometryRect> results;

  auto violation_position_set = layout->get_layout_engine()->copyPolyset();  /// copy polyset
  violation_position_set.clean();                                            /// eliminate overlaps
  ieda_solver::growAnd(violation_position_set, half_min_spacing);
  violation_position_set.get(results);
  for (auto& rect : results) {
    get_new_interval(ieda_solver::HORIZONTAL, rect);
    get_new_interval(ieda_solver::VERTICAL, rect);

    addViolation(rect, layer, ViolationEnumType::kDefaultSpacing);
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Spacing:\t") << results.size() << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc