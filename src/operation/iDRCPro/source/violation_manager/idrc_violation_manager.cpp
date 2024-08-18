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

#include "DRCViolationType.h"
#include "idrc_engine_manager.h"

namespace idrc {

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

void DrcViolationManager::set_net_ids(DrcEngineManager* engine_manager)
{
  for (auto& [type, violation_list] : _violation_list) {
#pragma omp parallel for
    for (auto* violation : violation_list) {
      if (violation == nullptr) {
        continue;
      }
      auto* violation_rect = static_cast<DrcViolationRect*>(violation);
      if (violation_rect->get_net_ids().size() <= 0) {
        auto layer = violation_rect->get_layer()->get_name();
        auto* layout = engine_manager->get_layout(layer);
        if (layout != nullptr) {
          auto net_ids = layout->querySubLayoutNetId(violation_rect->get_llx(), violation_rect->get_lly(), violation_rect->get_urx(),
                                                     violation_rect->get_ury());
          violation_rect->set_net_ids(net_ids);
        }
      }
    }
  }
}

std::map<ViolationEnumType, std::vector<DrcViolation*>> DrcViolationManager::get_violation_map(DrcEngineManager* engine_manager)
{
  set_net_ids(engine_manager);
  return std::move(_violation_list);
}

std::vector<DrcViolation*>& DrcViolationManager::get_violation_list(ViolationEnumType type)
{
  if (false == _violation_list.contains(type)) {
    _violation_list[type] = std::vector<DrcViolation*>{};
  }
  return _violation_list[type];
}

void DrcViolationManager::addViolation(int llx, int lly, int urx, int ury, ViolationEnumType type, std::set<int> net_id,
                                       std::string layer_name)
{
  idb::IdbLayer* layer = DrcTechRuleInst->findLayer(layer_name);
  DrcViolationRect* violation_rect = new DrcViolationRect(layer, type, llx, lly, urx, ury);
  violation_rect->set_net_ids(net_id);
  auto& violation_list = get_violation_list(type);
  violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
}

}  // namespace idrc