#pragma once

#include <string>

#include "Logger.hpp"

namespace irt {

enum class Orientation
{
  kNone = 0,
  kEast = 1,
  kWest = 2,
  kSouth = 3,
  kNorth = 4,
  kUp = 5,
  kDown = 6,
  kOblique = 7
};

struct GetOrientationName
{
  std::string operator()(const Orientation& orientation) const
  {
    std::string orientation_name;
    switch (orientation) {
      case Orientation::kNone:
        orientation_name = "none";
        break;
      case Orientation::kEast:
        orientation_name = "east";
        break;
      case Orientation::kWest:
        orientation_name = "west";
        break;
      case Orientation::kSouth:
        orientation_name = "south";
        break;
      case Orientation::kNorth:
        orientation_name = "north";
        break;
      case Orientation::kUp:
        orientation_name = "up";
        break;
      case Orientation::kDown:
        orientation_name = "down";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return orientation_name;
  }
};

}  // namespace irt
