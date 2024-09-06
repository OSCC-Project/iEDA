/**
 * @file feature_timing_eval.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-09-05
 * @brief feature evaluation with timing & power
 */

#include <algorithm>
#include <ranges>

#include "feature_builder.h"
#include "timing_api.hh"
#include "timing_db.hh"
namespace ieda_feature {
#define EVAL_STA_API_INST (ieval::TimingAPI::getInst())
TimingEvalSummary FeatureBuilder::buildTimingEvalSummary(const std::string& routing_type)
{
  ieval::TimingAPI::initRoutingType(routing_type);
  TimingEvalSummary timing_eval_summary;
  auto timing_summary = EVAL_STA_API_INST->evalDesign();
  std::ranges::for_each(timing_summary.clock_timings, [&timing_eval_summary](const auto& clock_timing) {
    timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.wns, clock_timing.tns, clock_timing.suggest_freq});
  });
  timing_eval_summary.power_info = {timing_summary.static_power, timing_summary.dynamic_power};
  return timing_eval_summary;
}
}  // namespace ieda_feature