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

void TimingEval::initRoutingType(const std::string& routing_type)
{
  RoutingType type = routing_type == "WLM"     ? RoutingType::kWLM
                     : routing_type == "HPWL"  ? RoutingType::kHPWL
                     : routing_type == "FLUTE" ? RoutingType::kFLUTE
                     : routing_type == "EGR"   ? RoutingType::kEGR
                     : routing_type == "DR"    ? RoutingType::kDR
                                               : RoutingType::kNone;
  InitSTA::initRoutingType(type);
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

TimingSummary TimingEval::evalDesign()
{
  TimingSummary summary;
  auto timing_map = EVAL_INIT_STA_INST->getTiming();
  for (const auto& [clock_name, timing_info] : timing_map) {
    ClockTiming clock_timing;
    clock_timing.clock_name = clock_name;
    clock_timing.wns = timing_info.at("WNS");
    clock_timing.tns = timing_info.at("TNS");
    clock_timing.suggest_freq = timing_info.at("Freq(MHz)");
    summary.timing.push_back(clock_timing);
  }
  auto power_map = EVAL_INIT_STA_INST->getPower();
  summary.static_power = power_map.at("static_power");
  summary.dynamic_power = power_map.at("dynamic_power");
  return summary;
}

double TimingEval::evalNetPower(const std::string& net_name) const
{
  return EVAL_INIT_STA_INST->evalNetPower(net_name);
}
std::map<std::string, double> TimingEval::evalAllNetPower() const
{
  return EVAL_INIT_STA_INST->evalAllNetPower();
}
}  // namespace ieval