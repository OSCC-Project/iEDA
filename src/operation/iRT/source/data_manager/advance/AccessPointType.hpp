#pragma once

#include "Logger.hpp"

namespace irt {

enum class AccessPointType
{
  kNone = 0,
  kPrefTrackGrid = 1,
  kPrefTrackCenter = 2,
  kShapeCenter = 3
};

struct GetAccessPointTypeName
{
  std::string operator()(const AccessPointType& access_point_type) const
  {
    std::string access_point_name;
    switch (access_point_type) {
      case AccessPointType::kNone:
        access_point_name = "none";
        break;
      case AccessPointType::kPrefTrackGrid:
        access_point_name = "pref_track_grid";
        break;
      case AccessPointType::kPrefTrackCenter:
        access_point_name = "pref_track_center";
        break;
      case AccessPointType::kShapeCenter:
        access_point_name = "shape_center";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return access_point_name;
  }
};

}  // namespace irt
