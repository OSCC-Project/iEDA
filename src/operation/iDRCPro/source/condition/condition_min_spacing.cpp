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
  ieda::Stats states;
  int min_spacing_count = 0;
  auto& layer_polyset = layout->get_layout()->get_engine()->get_polyset();
  int min_spacing = DrcTechRuleInst->getMinSpacing(layer);
  if (min_spacing > 0) {
    auto violation_position_set = layer_polyset;
    int half_min_spacing = min_spacing / 2;
    gtl::grow_and(violation_position_set, half_min_spacing);
    std::vector<ieda_solver::GeometryRect> results;
    violation_position_set.get(results);

    auto get_new_interval = [&](ieda_solver::GeometryOrientation direction, ieda_solver::GeometryRect& rect) {
      int length = ieda_solver::getWireWidth(rect, direction);
      if (length <= half_min_spacing) {
        int expand_length = std::abs(half_min_spacing - length);
        ieda_solver::bloat(rect, direction, expand_length);
      } else if (length > min_spacing) {
        ieda_solver::shrink(rect, direction, half_min_spacing);
      } else {
        int shrink_length = std::abs(half_min_spacing - length);
        ieda_solver::shrink(rect, direction, shrink_length);
      }
    };

    for (auto& rect : results) {
      get_new_interval(ieda_solver::HORIZONTAL, rect);
      get_new_interval(ieda_solver::VERTICAL, rect);
      addViolation(rect, layer, ViolationEnumType::kDefaultSpacing);
    }
    min_spacing_count = results.size();
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Spacing:\t") << min_spacing_count << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc