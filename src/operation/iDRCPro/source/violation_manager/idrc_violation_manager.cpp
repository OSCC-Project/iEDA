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
#include "DRCViolationType.h"

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

}  // namespace idrc