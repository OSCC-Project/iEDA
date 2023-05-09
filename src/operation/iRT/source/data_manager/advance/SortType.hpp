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

#include <string>

#include "Logger.hpp"

namespace irt {

enum class SortType
{
  kNone,
  kClockPriority,
  kRoutingAreaASC,
  kLengthWidthRatioDESC,
  kPinNumDESC
};

struct GetSortTypeName
{
  std::string operator()(const SortType& sort_type) const
  {
    std::string sort_name;
    switch (sort_type) {
      case SortType::kNone:
        sort_name = "none";
        break;
      case SortType::kClockPriority:
        sort_name = "clock_priority";
        break;
      case SortType::kRoutingAreaASC:
        sort_name = "routing_area_asc";
        break;
      case SortType::kLengthWidthRatioDESC:
        sort_name = "length_width_ratio_desc";
        break;
      case SortType::kPinNumDESC:
        sort_name = "pin_num_desc";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return sort_name;
  }
};

}  // namespace irt
