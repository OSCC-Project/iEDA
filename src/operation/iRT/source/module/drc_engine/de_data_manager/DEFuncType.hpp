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

enum class DEFuncType
{
  kNone,
  kGetDEProcessType,
  kExpandViolation
};

struct GetDEFuncTypeName
{
  std::string operator()(const DEFuncType& process_type) const
  {
    std::string process_type_name;
    switch (process_type) {
      case DEFuncType::kNone:
        process_type_name = "none";
        break;
      case DEFuncType::kGetDEProcessType:
        process_type_name = "get_de_process_type";
        break;
      case DEFuncType::kExpandViolation:
        process_type_name = "expand_violation";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return process_type_name;
  }
};

}  // namespace irt
