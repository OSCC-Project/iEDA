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
