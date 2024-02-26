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
#include "idrc_violation.h"

namespace idrc {
// class DrcViolation;

class DrcViolationManager
{
 public:
  DrcViolationManager();
  ~DrcViolationManager();

  std::map<ViolationEnumType, std::vector<DrcViolation*>> get_violation_map() { return std::move(_violation_list); }

  std::vector<DrcViolation*>& get_violation_list(ViolationEnumType type)
  {
    if (false == _violation_list.contains(type)) {
      _violation_list[type] = std::vector<DrcViolation*>{};
    }

    return _violation_list[type];
  }

  void addViolation(int llx, int lly, int urx, int ury, ViolationEnumType type, std::set<int> net_id, std::string layer_name)
  {
    // todo
    auto idb_design = dmInst->get_idb_design();
    auto idb_layout = idb_design->get_layout();

    idb::IdbLayer* layer = idb_layout->get_layers()->find_layer(layer_name);

    DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_id, type, llx, lly, urx, ury);
    auto& violation_list = get_violation_list(type);
    violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
  }

 private:
  std::map<ViolationEnumType, std::vector<DrcViolation*>> _violation_list;
};

}  // namespace idrc