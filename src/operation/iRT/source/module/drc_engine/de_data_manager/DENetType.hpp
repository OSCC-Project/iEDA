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

enum class DENetType
{
  kNone,
  kSingleNet,
  kMultiNet
};

struct GetDENetTypeName
{
  std::string operator()(const DENetType& net_type) const
  {
    std::string net_type_name;
    switch (net_type) {
      case DENetType::kNone:
        net_type_name = "none";
        break;
      case DENetType::kSingleNet:
        net_type_name = "single_net";
        break;
      case DENetType::kMultiNet:
        net_type_name = "multi_net";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return net_type_name;
  }
};

}  // namespace irt
