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

TimingEval* TimingEval::getInst()
{
  if (_timing_eval == nullptr) {
    _timing_eval = new TimingEval();
  }
  return _timing_eval;
}

void TimingEval::runSTA()
{
  EVAL_INIT_STA_INST->runSTA();
}

void TimingEval::runVecSTA(ivec::VecLayout* vec_layout)
{
  EVAL_INIT_STA_INST->runVecSTA(vec_layout, "./");
}

void TimingEval::evalTiming(const std::string& routing_type, const bool& rt_done)
{
  EVAL_INIT_STA_INST->evalTiming(routing_type, rt_done);
}

// for vectorization(to weiguo)
TimingWireGraph* TimingEval::getTimingWireGraph()
{
  auto timing_wire_graph = EVAL_INIT_STA_INST->getTimingWireGraph();
  auto timing_wire_graph_ptr = new TimingWireGraph(std::move(timing_wire_graph));
  return timing_wire_graph_ptr;
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

double TimingEval::getEarlySlack(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getEarlySlack(pin_name);
}

double TimingEval::getLateSlack(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getLateSlack(pin_name);
}

double TimingEval::getArrivalEarlyTime(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getArrivalEarlyTime(pin_name);
}

double TimingEval::getArrivalLateTime(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getArrivalLateTime(pin_name);
}

double TimingEval::getRequiredEarlyTime(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getRequiredEarlyTime(pin_name);
}

double TimingEval::getRequiredLateTime(const std::string& pin_name) const
{
  return EVAL_INIT_STA_INST->getRequiredLateTime(pin_name);
}

double TimingEval::reportWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return EVAL_INIT_STA_INST->reportWNS(clock_name, mode);
}

double TimingEval::reportTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return EVAL_INIT_STA_INST->reportTNS(clock_name, mode);
}

void TimingEval::updateTiming(const std::vector<TimingNet*>& timing_net_list, int32_t dbu_unit)
{
  EVAL_INIT_STA_INST->updateTiming(timing_net_list, dbu_unit);
}

void TimingEval::updateTiming(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                              const int& propagation_level, int32_t dbu_unit)
{
  EVAL_INIT_STA_INST->updateTiming(timing_net_list, name_list, propagation_level, dbu_unit);
}

bool TimingEval::isClockNet(const std::string& net_name) const
{
  return EVAL_INIT_STA_INST->isClockNet(net_name);
}

std::map<int, double> TimingEval::patchTimingMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  return EVAL_INIT_STA_INST->patchTimingMap(patch);
}

std::map<int, double> TimingEval::patchPowerMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  return EVAL_INIT_STA_INST->patchPowerMap(patch);
}

std::map<int, double> TimingEval::patchIRDropMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  return EVAL_INIT_STA_INST->patchIRDropMap(patch);
}

}  // namespace ieval