/**
 * @file timing_eval.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief evaluation with timing & power
 */

#include "timing_eval.hh"

#include "init_sta.hh"
namespace ieval {
#define EVAL_INIT_STA_INST (ieval::InitSTA::getInst())
TimingEval* TimingEval::_timing_eval = nullptr;

TimingEval::TimingEval()
{
  EVAL_INIT_STA_INST->runSTA();
}

TimingEval* TimingEval::getInst()
{
  if (_timing_eval == nullptr) {
    _timing_eval = new TimingEval();
  }
  return _timing_eval;
}

void TimingEval::destroyInst()
{
  delete _timing_eval;
  _timing_eval = nullptr;
}

std::map<std::string, TimingSummary> TimingEval::evalDesign()
{
  std::map<std::string, TimingSummary> summary;
  auto type_timing_map = EVAL_INIT_STA_INST->getTiming();
  auto type_power_map = EVAL_INIT_STA_INST->getPower();
  for (const auto& [routing_type, timing_map] : type_timing_map) {
    summary[routing_type] = TimingSummary();
    for (const auto& [clock_name, timing_info] : timing_map) {
      ClockTiming clock_timing;
      clock_timing.clock_name = clock_name;
      clock_timing.setup_tns = timing_info.at("setup_tns");
      clock_timing.setup_wns = timing_info.at("setup_wns");
      clock_timing.hold_tns = timing_info.at("hold_tns");
      clock_timing.hold_wns = timing_info.at("hold_wns");
      clock_timing.suggest_freq = timing_info.at("suggest_freq");
      summary[routing_type].clock_timings.push_back(clock_timing);
    }
    summary[routing_type].static_power = type_power_map[routing_type].at("static_power");
    summary[routing_type].dynamic_power = type_power_map[routing_type].at("dynamic_power");
  }
  return summary;
}

std::map<std::string, std::unordered_map<std::string, double>> TimingEval::evalNetPower() const
{
  return EVAL_INIT_STA_INST->getNetPower();
}
}  // namespace ieval