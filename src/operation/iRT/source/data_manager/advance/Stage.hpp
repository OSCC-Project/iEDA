#pragma once

#include "Logger.hpp"

namespace irt {

enum class Stage
{
  kNone,
  kDetailedRouter,
  kGlobalRouter,
  kPinAccessor,
  kResourceAllocator,
  kTrackAssigner,
  kViolationRepairer
};

struct GetStageName
{
  std::string operator()(const Stage& stage) const
  {
    std::string stage_name;
    switch (stage) {
      case Stage::kNone:
        stage_name = "none";
        break;
      case Stage::kDetailedRouter:
        stage_name = "detailed_router";
        break;
      case Stage::kGlobalRouter:
        stage_name = "global_router";
        break;
      case Stage::kPinAccessor:
        stage_name = "pin_accessor";
        break;
      case Stage::kResourceAllocator:
        stage_name = "resource_allocator";
        break;
      case Stage::kTrackAssigner:
        stage_name = "track_assigner";
        break;
      case Stage::kViolationRepairer:
        stage_name = "violation_repairer";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return stage_name;
  }
};

}  // namespace irt
