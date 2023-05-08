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
