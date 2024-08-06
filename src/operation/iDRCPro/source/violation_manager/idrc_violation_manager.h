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
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "DRCViolationType.h"
#include "IdbLayer.h"
#include "boost_definition.h"
#include "idm.h"
#include "idrc_dm.h"
#include "idrc_region_query.h"
#include "idrc_violation.h"
#include "tech_rules.h"

namespace idrc {
// class DrcViolation;

class DrcViolationManager
{
 public:
  DrcViolationManager(DrcDataManager* data_manager) : _data_manager(data_manager){};
  ~DrcViolationManager();

  void get_net_id()
  {
    for (auto& [type, violation_list] : _violation_list) {
      for (auto* violation : violation_list) {
        auto* violation_rect = static_cast<DrcViolationRect*>(violation);
        auto net_ids = _data_manager->get_region_query()->queryNetId(violation_rect->get_layer()->get_name(), violation_rect->get_llx(),
                                                                     violation_rect->get_lly(), violation_rect->get_urx(),
                                                                     violation_rect->get_ury());
        violation_rect->set_net_ids(net_ids);
      }
    }
  }

  std::map<ViolationEnumType, std::vector<DrcViolation*>> get_violation_map()
  {
    get_net_id();
    return std::move(_violation_list);
  }

  std::vector<DrcViolation*>& get_violation_list(ViolationEnumType type)
  {
    if (false == _violation_list.contains(type)) {
      _violation_list[type] = std::vector<DrcViolation*>{};
    }

    return _violation_list[type];
  }

  void addViolation(int llx, int lly, int urx, int ury, ViolationEnumType type, std::set<int> net_id, std::string layer_name)
  {
    idb::IdbLayer* layer = DrcTechRuleInst->findLayer(layer_name);

    DrcViolationRect* violation_rect = new DrcViolationRect(layer, type, llx, lly, urx, ury);
    violation_rect->set_net_ids(net_id);
    auto& violation_list = get_violation_list(type);
    violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
  }

 private:
  DrcDataManager* _data_manager = nullptr;
  std::map<ViolationEnumType, std::vector<DrcViolation*>> _violation_list;
};

}  // namespace idrc