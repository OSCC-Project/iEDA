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

enum class ConnectType
{
  kNone = 0,
  kSignal = 1,
  kPower = 2,
  kGround = 3,
  kClock = 4,
  kAnalog = 5,
  kReset = 6,
  kScan = 7,
  kTieoff = 8
};

struct GetConnectTypeName
{
  std::string operator()(const ConnectType& connect_type) const
  {
    std::string connect_name;
    switch (connect_type) {
      case ConnectType::kNone:
        connect_name = "none";
        break;
      case ConnectType::kSignal:
        connect_name = "signal";
        break;
      case ConnectType::kPower:
        connect_name = "power";
        break;
      case ConnectType::kGround:
        connect_name = "ground";
        break;
      case ConnectType::kClock:
        connect_name = "clock";
        break;
      case ConnectType::kAnalog:
        connect_name = "analog";
        break;
      case ConnectType::kReset:
        connect_name = "reset";
        break;
      case ConnectType::kScan:
        connect_name = "scan";
        break;
      case ConnectType::kTieoff:
        connect_name = "tieoff";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return connect_name;
  }
};

}  // namespace irt
