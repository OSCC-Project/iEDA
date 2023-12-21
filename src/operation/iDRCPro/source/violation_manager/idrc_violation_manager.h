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
#include <string>
#include <vector>

#include "BaseRegion.hpp"
#include "BaseShape.hpp"
#include "BaseViolationInfo.hpp"
#include "boost_definition.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"

namespace idrc {
// class DrcViolation;

class DrcViolationManager
{
 public:
  DrcViolationManager();
  ~DrcViolationManager();

  std::map<std::string, std::vector<irt::BaseViolationInfo>> get_rt_violation_map();
  std::map<ViolationEnumType, std::vector<DrcViolation*>> get_violation_map() { return std::move(_violation_list); }

  std::vector<DrcViolation*>& get_violation_list(ViolationEnumType type)
  {
    if (false == _violation_list.contains(type)) {
      _violation_list[type] = std::vector<DrcViolation*>{};
    }

    return _violation_list[type];
  }

  /// debug
  std::vector<ieda_solver::GtlRect> get_boost_rects(idb::IdbLayer* layer)
  {
    std::vector<ieda_solver::GtlRect> boost_rects;

    for (auto& violation : _violation_list) {
      for (auto& violation_shape : violation.second) {
        if (violation_shape->get_layer() == layer) {
          if (violation_shape->is_rect()) {
            auto* rect = static_cast<DrcViolationRect*>(violation_shape);
            ieda_solver::GtlRect boost_rect(rect->get_llx(), rect->get_lly(), rect->get_urx(), rect->get_ury());
            boost_rects.emplace_back(boost_rect);
          }
        }
      }
    }

    return boost_rects;
  }

 private:
  std::map<ViolationEnumType, std::vector<DrcViolation*>> _violation_list;
};

}  // namespace idrc