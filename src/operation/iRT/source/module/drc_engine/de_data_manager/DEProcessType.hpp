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

#include "Logger.hpp"

namespace irt {

enum class DEProcessType
{
  kNone,
  kSkip,
  kCost,
  kPatch
};

struct GetDEProcessTypeName
{
  std::string operator()(const DEProcessType& process_type) const
  {
    std::string process_type_name;
    switch (process_type) {
      case DEProcessType::kNone:
        process_type_name = "none";
        break;
      case DEProcessType::kSkip:
        process_type_name = "skip";
        break;
      case DEProcessType::kCost:
        process_type_name = "cost";
        break;
      case DEProcessType::kPatch:
        process_type_name = "patch";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return process_type_name;
  }
};

}  // namespace irt
