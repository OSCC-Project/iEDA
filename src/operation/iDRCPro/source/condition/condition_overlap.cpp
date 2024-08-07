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

void DrcConditionManager::checkOverlap(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kShort) == _check_select.end()) {
    return;
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Metal Short:\t") << " Begin checking drc for layer " << layer);

  ieda::Stats states;
  int total = 0;
  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    auto [llx, lly, urx, ury] = sub_layout->get_engine()->bounding_box();

    auto query_sub_layouts = layout->querySubLayouts(llx, lly, urx, ury);
    int check_size = 0;
    for (auto* query_sub_layout : query_sub_layouts) {
      auto query_id = query_sub_layout->get_id();
      if (query_id == -1 || query_id == net_id || true == sub_layout->hasCheckIntersect(query_id)) {
        continue;
      }
      auto& overlaps = sub_layout->get_engine()->getOverlap(query_sub_layout->get_engine());
      std::set<int> net_ids = {};
      if (overlaps.size() > 0) {
        net_ids.insert(net_id);
        net_ids.insert(query_id);
      }
      for (auto& overlap_polygon : overlaps) {
        ieda_solver::GeometryRect overlap_violation_rect;
        ieda_solver::envelope(overlap_violation_rect, overlap_polygon);
        addViolation(overlap_violation_rect, layer, ViolationEnumType::kShort, net_ids);
      }

      total += overlaps.size();
      sub_layout->addIntersectNet(query_id);
      query_sub_layout->addIntersectNet(net_id);
      check_size++;
    }
#ifdef DEBUGCLOSE_OVERLAP
    DEBUGOUTPUT(DEBUGHIGHLIGHT("net_id:\t") << net_id << "\tlayer " << layer << "\tllx = " << llx << "\tlly = " << lly << "\turx = " << urx
                                            << "\tllx = " << ury << "\tquery_sub_layouts = " << query_sub_layouts.size()
                                            << "\tcheck size = " << check_size << "\ttotal = " << total);
#else

#endif
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Metal Short:\t") << total << "\tlayer " << layer << "\tnets = " << layout->get_sub_layouts().size()
                                               << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc