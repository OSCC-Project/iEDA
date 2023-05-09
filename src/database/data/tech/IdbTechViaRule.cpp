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
#include "IdbTechViaRule.h"

namespace idb {
  void IdbTechViaRuleList::printViaRule() {
    for (auto &viaRule : _tech_via_rules) {
      std::cout << "viaRuleGenerateName ::" << viaRule->get_name() << std::endl;
    }
  }

  IdbTechViaRule *IdbTechViaRuleList::getTechViaRule(int cutLayerId) {
    for (auto &viaRule : _tech_via_rules) {
      if (viaRule->get_cut_layer_id() == cutLayerId) {
        return viaRule.get();
      }
    }
    return nullptr;
  }
}  // namespace idb
