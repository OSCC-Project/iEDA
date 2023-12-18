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

#include <iostream>
#include <string>

namespace irt {
/**
 * ┌──┬─────────┬─────────┬─────────┐
 * │  │ Spacing │ MinArea │ MinStep │
 * ├──┼─────────┼─────────┼─────────┤
 * │PA│    Y    │    X    │    X    │
 * ├──┼─────────┼─────────┼─────────┤
 * │GR│    Y    │    X    │    X    │
 * ├──┼─────────┼─────────┼─────────┤
 * │TA│    Y    │    X    │    X    │
 * ├──┼─────────┼─────────┼─────────┤
 * │DR│    Y    │    X    │    X    │
 * ├──┼─────────┼─────────┼─────────┤
 * │VR│    Y    │    Y    │    Y    │
 * └──┴─────────┴─────────┴─────────┘
 */
enum class DRCCheckType
{
  kNone,
  kSpacing,
  kMinArea,
  kMinStep
};

struct GetDRCCheckTypeName
{
  std::string operator()(const DRCCheckType& dr_source_type) const
  {
    std::string check_type_name;
    switch (dr_source_type) {
      case DRCCheckType::kNone:
        check_type_name = "none";
        break;
      case DRCCheckType::kSpacing:
        check_type_name = "spacing";
        break;
      case DRCCheckType::kMinArea:
        check_type_name = "min_area";
        break;
      case DRCCheckType::kMinStep:
        check_type_name = "min_step";
        break;
      default:
        std::cout << "check_type is error!" << std::endl;
        break;
    }
    return check_type_name;
  }
};

}  // namespace irt
