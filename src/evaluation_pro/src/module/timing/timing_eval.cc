/**
 * @file timing_eval.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief evaluation with timing & power
 */

#include "timing_eval.hh"

namespace ieval {

TimingSummary TimingEval::evalDesign()
{
  TimingSummary summary;
  auto timing_map = _init_sta->getTiming();
  for (const auto& [clock_name, timing_info] : timing_map) {
    ClockTiming clock_timing;
    clock_timing.clock_name = clock_name;
    clock_timing.wns = timing_info.at("WNS");
    clock_timing.tns = timing_info.at("TNS");
    clock_timing.suggest_freq = timing_info.at("Freq(MHz)");
    summary.timing.push_back(clock_timing);
  }
  auto power_map = _init_sta->getPower();
  summary.static_power = power_map.at("static_power");
  summary.dynamic_power = power_map.at("dynamic_power");
  return summary;
}

double TimingEval::evalNetPower(const std::string& net_name) const
{
  return _init_sta->evalNetPower(net_name);
}
std::map<std::string, double> TimingEval::evalAllNetPower() const
{
  return _init_sta->evalAllNetPower();
}
}  // namespace ieval