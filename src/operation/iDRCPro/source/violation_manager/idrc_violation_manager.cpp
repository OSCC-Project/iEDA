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
#include "idrc_violation_manager.h"

// #include "idrc_violation.h"
#include "idrc_violation_enum.h"

namespace idrc {
DrcViolationManager::DrcViolationManager()
{
}

DrcViolationManager::~DrcViolationManager()
{
  for (auto& [type, violations] : _violation_list) {
    for (auto* violation : violations) {
      if (violation != nullptr) {
        delete violation;
        violation = nullptr;
      }
    }

    violations.clear();
    std::vector<DrcViolation*>().swap(violations);
  }
  _violation_list.clear();
}

std::map<std::string, std::vector<irt::BaseViolationInfo>> DrcViolationManager::get_rt_violation_map()
{
  std::map<std::string, std::vector<irt::BaseViolationInfo>> rt_violation_map;

  for (auto& [type, violations] : _violation_list) {
    if (violations.size() <= 0) {
      continue;
    }

    std::string type_name = DrcViolationTypeInst->get_type_name(type);
    std::vector<irt::BaseViolationInfo> rt_violations;
    rt_violations.reserve(violations.size());
    for (auto* violation : violations) {
      if (violation->is_rect()) {
        auto* rect = static_cast<DrcViolationRect*>(violation);
        irt::BaseViolationInfo violation;
        violation.set_rule_name(type_name);
        violation.set_violation_region(rect->get_llx(), rect->get_lly(), rect->get_urx(), rect->get_ury());
        violation.set_layer_idx(rect->get_layer_id());

        std::set<irt::BaseInfo, irt::CmpBaseInfo> base_info_set;
        for (auto net_id : rect->get_net_ids()) {
          irt::BaseInfo info;
          info.set_net_idx(net_id);
          base_info_set.emplace(info);
        }

        violation.set_base_info_set(base_info_set);
        rt_violations.emplace_back(violation);
      }
    }

    rt_violation_map[type_name] = rt_violations;
  }

  return rt_violation_map;
}

}  // namespace idrc