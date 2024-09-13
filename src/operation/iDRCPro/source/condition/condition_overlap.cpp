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
#include "omp.h"

namespace idrc {

void DrcConditionManager::checkOverlap(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kShort) == _check_select.end()) {
    return;
  }

  ieda::Stats states;

  struct DrcShortInfo
  {
    ieda_solver::GeometryRect rect;
    std::set<int> net_ids;
  };

  std::map<int, std::vector<DrcShortInfo>> drc_map;  /// save violation for each sublayout
  std::vector<DrcEngineSubLayout*> sub_layouts;
  /// init sublayout drc map & sublayout list
  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    drc_map.insert(std::make_pair(net_id, std::vector<DrcShortInfo>{}));
    sub_layouts.push_back(sub_layout);
  }

/// lock-free parallel
#pragma omp parallel for schedule(dynamic)
  for (auto sub_layout : sub_layouts) {
    auto net_id = sub_layout->get_id();

    /// skip environment checking for RT result
    if (_check_type == DrcCheckerType::kRT && net_id < 0) {
      continue;
    }
    /// VDD & VSS must be check for def
    if (net_id < 0 && (net_id != NET_ID_VDD || net_id != NET_ID_VSS)) {
      continue;
    }

    if (sub_layout == nullptr) {
      continue;
    }

    std::vector<DrcShortInfo> this_drc_list;  /// violation for this sublayout

    int checking_size = 0;
    for (auto& [query_id, query_sub_layout] : sub_layout->get_intersect_layouts()) {
      if (query_id == net_id || true == sub_layout->hasChecked(query_id)) {
        continue;
      }

      auto overlaps = sub_layout->get_engine()->getOverlap(query_sub_layout->get_engine());
      std::set<int> net_ids = {};
      if (overlaps.size() > 0) {
        net_ids.insert(net_id);
        net_ids.insert(query_id);
      }

      for (auto& overlap_polygon : overlaps) {
        ieda_solver::GeometryRect overlap_violation_rect;
        ieda_solver::envelope(overlap_violation_rect, overlap_polygon);

        // addViolation(overlap_violation_rect, layer, ViolationEnumType::kShort, net_ids);
        DrcShortInfo drc_info;
        drc_info.rect = overlap_violation_rect;
        drc_info.net_ids = net_ids;

        this_drc_list.push_back(drc_info);
      }

      checking_size++;
    }

    drc_map[net_id] = this_drc_list;

    DEBUGOUTPUT(DEBUGHIGHLIGHT("net_id:\t") << net_id << "\tlayer " << layer << "\tchecking_size = " << checking_size
                                            << "\toverlaps = " << this_drc_list.size());
  }

  int total_drc = 0;
  for (auto& [net_id, drc_list] : drc_map) {
    for (auto drc : drc_list) {
      addViolation(drc.rect, layer, ViolationEnumType::kShort, drc.net_ids);
      total_drc++;
    }
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Metal Short:\t") << total_drc << "\tlayer " << layer << "\tnets = " << layout->get_sub_layouts().size()
                                               << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc