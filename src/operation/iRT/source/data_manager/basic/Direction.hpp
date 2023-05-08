#pragma once

#include "Logger.hpp"

namespace irt {

enum class Direction
{
  kNone = 0,
  kHorizontal = 1,
  kVertical = 2,
  kOblique = 3,
  kProximal = 4
};

struct GetDirectionName
{
  std::string operator()(const Direction& direction) const
  {
    std::string direction_name;
    switch (direction) {
      case Direction::kNone:
        direction_name = "none";
        break;
      case Direction::kHorizontal:
        direction_name = "horizontal";
        break;
      case Direction::kVertical:
        direction_name = "vertical";
        break;
      case Direction::kOblique:
        direction_name = "oblique";
        break;
      case Direction::kProximal:
        direction_name = "proximal";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return direction_name;
  }
};

}  // namespace irt
