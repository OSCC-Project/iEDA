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

TimingEvalSummary FeatureBuilder::buildTimingEvalSummary()
{
  TimingEvalSummary timing_eval_summary;
  EVAL_STA_API_INST->runSTA();
  auto timing_summary = EVAL_STA_API_INST->evalDesign();
  // TBD
  // WlmTimingEvalSummary wlm_timing_eval_summary;
  // auto wlm_timing_summary = timing_summary.at("WLM");
  // std::ranges::for_each(wlm_timing_summary.clock_timings, [&wlm_timing_eval_summary](const auto& clock_timing) {
  //   wlm_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
  //                                                    clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  // });
  // wlm_timing_eval_summary.power_info = {wlm_timing_summary.static_power, wlm_timing_summary.dynamic_power};
  // timing_eval_summary.wlm_timing_eval_summary = wlm_timing_eval_summary;

  HpwlTimingEvalSummary hpwl_timing_eval_summary;
  auto hpwl_timing_summary = timing_summary.at("HPWL");
  std::ranges::for_each(hpwl_timing_summary.clock_timings, [&hpwl_timing_eval_summary](const auto& clock_timing) {
    hpwl_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                      clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  });
  hpwl_timing_eval_summary.power_info = {hpwl_timing_summary.static_power, hpwl_timing_summary.dynamic_power};
  timing_eval_summary.hpwl_timing_eval_summary = hpwl_timing_eval_summary;

  FluteTimingEvalSummary flute_timing_eval_summary;
  auto flute_timing_summary = timing_summary.at("FLUTE");
  std::ranges::for_each(flute_timing_summary.clock_timings, [&flute_timing_eval_summary](const auto& clock_timing) {
    flute_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                       clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  });
  flute_timing_eval_summary.power_info = {flute_timing_summary.static_power, flute_timing_summary.dynamic_power};
  timing_eval_summary.flute_timing_eval_summary = flute_timing_eval_summary;

  SaltTimingEvalSummary salt_timing_eval_summary;
  auto salt_timing_summary = timing_summary.at("SALT");
  std::ranges::for_each(salt_timing_summary.clock_timings, [&salt_timing_eval_summary](const auto& clock_timing) {
    salt_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                      clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  });
  salt_timing_eval_summary.power_info = {salt_timing_summary.static_power, salt_timing_summary.dynamic_power};
  timing_eval_summary.salt_timing_eval_summary = salt_timing_eval_summary;

  EgrTimingEvalSummary egr_timing_eval_summary;
  auto egr_timing_summary = timing_summary.at("EGR");
  std::ranges::for_each(egr_timing_summary.clock_timings, [&egr_timing_eval_summary](const auto& clock_timing) {
    egr_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                     clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  });
  egr_timing_eval_summary.power_info = {egr_timing_summary.static_power, egr_timing_summary.dynamic_power};
  timing_eval_summary.egr_timing_eval_summary = egr_timing_eval_summary;

  DrTimingEvalSummary dr_timing_eval_summary;
  auto dr_timing_summary = timing_summary.at("DR");
  std::ranges::for_each(dr_timing_summary.clock_timings, [&dr_timing_eval_summary](const auto& clock_timing) {
    dr_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                    clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
  });
  dr_timing_eval_summary.power_info = {dr_timing_summary.static_power, dr_timing_summary.dynamic_power};
  timing_eval_summary.dr_timing_eval_summary = dr_timing_eval_summary;

  return timing_eval_summary;
}
TimingEvalSummary FeatureBuilder::buildTimingUnionEvalSummary()
{
  TimingEvalSummary timing_eval_summary;
  auto timing_summary = EVAL_STA_API_INST->evalDesign();
  if (timing_summary.contains("WLM")) {
    WlmTimingEvalSummary wlm_timing_eval_summary;
    auto wlm_timing_summary = timing_summary.at("WLM");
    std::ranges::for_each(wlm_timing_summary.clock_timings, [&wlm_timing_eval_summary](const auto& clock_timing) {
      wlm_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                       clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    wlm_timing_eval_summary.power_info = {wlm_timing_summary.static_power, wlm_timing_summary.dynamic_power};
    timing_eval_summary.wlm_timing_eval_summary = wlm_timing_eval_summary;
  }
  if (timing_summary.contains("HPWL")) {
    HpwlTimingEvalSummary hpwl_timing_eval_summary;
    auto hpwl_timing_summary = timing_summary.at("HPWL");
    std::ranges::for_each(hpwl_timing_summary.clock_timings, [&hpwl_timing_eval_summary](const auto& clock_timing) {
      hpwl_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                        clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    hpwl_timing_eval_summary.power_info = {hpwl_timing_summary.static_power, hpwl_timing_summary.dynamic_power};
    timing_eval_summary.hpwl_timing_eval_summary = hpwl_timing_eval_summary;
  }
  if (timing_summary.contains("FLUTE")) {
    FluteTimingEvalSummary flute_timing_eval_summary;
    auto flute_timing_summary = timing_summary.at("FLUTE");
    std::ranges::for_each(flute_timing_summary.clock_timings, [&flute_timing_eval_summary](const auto& clock_timing) {
      flute_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                         clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    flute_timing_eval_summary.power_info = {flute_timing_summary.static_power, flute_timing_summary.dynamic_power};
    timing_eval_summary.flute_timing_eval_summary = flute_timing_eval_summary;
  }
  if (timing_summary.contains("SALT")) {
    SaltTimingEvalSummary salt_timing_eval_summary;
    auto salt_timing_summary = timing_summary.at("SALT");
    std::ranges::for_each(salt_timing_summary.clock_timings, [&salt_timing_eval_summary](const auto& clock_timing) {
      salt_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                        clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    salt_timing_eval_summary.power_info = {salt_timing_summary.static_power, salt_timing_summary.dynamic_power};
    timing_eval_summary.salt_timing_eval_summary = salt_timing_eval_summary;
  }
  if (timing_summary.contains("EGR")) {
    EgrTimingEvalSummary egr_timing_eval_summary;
    auto egr_timing_summary = timing_summary.at("EGR");
    std::ranges::for_each(egr_timing_summary.clock_timings, [&egr_timing_eval_summary](const auto& clock_timing) {
      egr_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                       clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    egr_timing_eval_summary.power_info = {egr_timing_summary.static_power, egr_timing_summary.dynamic_power};
    timing_eval_summary.egr_timing_eval_summary = egr_timing_eval_summary;
  }
  if (timing_summary.contains("DR")) {
    DrTimingEvalSummary dr_timing_eval_summary;
    auto dr_timing_summary = timing_summary.at("DR");
    std::ranges::for_each(dr_timing_summary.clock_timings, [&dr_timing_eval_summary](const auto& clock_timing) {
      dr_timing_eval_summary.clock_timings.push_back({clock_timing.clock_name, clock_timing.setup_tns, clock_timing.setup_wns,
                                                      clock_timing.hold_tns, clock_timing.hold_wns, clock_timing.suggest_freq});
    });
    dr_timing_eval_summary.power_info = {dr_timing_summary.static_power, dr_timing_summary.dynamic_power};
    timing_eval_summary.dr_timing_eval_summary = dr_timing_eval_summary;
  }
  return timing_eval_summary;
}
void FeatureBuilder::evalTiming(const std::string& routing_type, const bool& rt_done)
{
  EVAL_STA_API_INST->evalTiming(routing_type, rt_done);
}
}  // namespace ieda_feature