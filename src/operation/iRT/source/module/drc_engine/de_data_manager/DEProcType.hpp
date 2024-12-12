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

enum class DEProcType
{
  kNone,
  kIgnore,
  kGet
};

struct GetDEProcTypeName
{
  std::string operator()(const DEProcType& proc_type) const
  {
    std::string proc_type_name;
    switch (proc_type) {
      case DEProcType::kNone:
        proc_type_name = "none";
        break;
      case DEProcType::kIgnore:
        proc_type_name = "ignore";
        break;
      case DEProcType::kGet:
        proc_type_name = "get";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return proc_type_name;
  }
};

}  // namespace irt
