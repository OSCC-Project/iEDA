/**
 * @file timing_api.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief api for timing & power evaluation
 */

#include "timing_api.hh"

#include "timing_eval.hh"

namespace ieval {

#define EVAL_STA_INST (ieval::TimingEval::getInst())

TimingAPI* TimingAPI::_timing_api = nullptr;

TimingAPI* TimingAPI::getInst()
{
  if (_timing_api == nullptr) {
    _timing_api = new TimingAPI();
  }
  return _timing_api;
}

void TimingAPI::runSTA()
{
  EVAL_STA_INST->runSTA();
}

void TimingAPI::evalTiming(const std::string& routing_type, const bool& rt_done)
{
  EVAL_STA_INST->evalTiming(routing_type, rt_done);
}

void TimingAPI::destroyInst()
{
  ieval::TimingEval::destroyInst();
}

std::map<std::string, TimingSummary> TimingAPI::evalDesign()
{
  return EVAL_STA_INST->evalDesign();
}
std::map<std::string, std::unordered_map<std::string, double>> TimingAPI::evalNetPower() const
{
  return EVAL_STA_INST->evalNetPower();
}

double TimingAPI::getEarlySlack(const std::string& pin_name) const
{
  return EVAL_STA_INST->getEarlySlack(pin_name);
}

double TimingAPI::getLateSlack(const std::string& pin_name) const
{
  return EVAL_STA_INST->getLateSlack(pin_name);
}

double TimingAPI::getArrivalEarlyTime(const std::string& pin_name) const
{
  return EVAL_STA_INST->getArrivalEarlyTime(pin_name);
}

double TimingAPI::getArrivalLateTime(const std::string& pin_name) const
{
  return EVAL_STA_INST->getArrivalLateTime(pin_name);
}

double TimingAPI::getRequiredEarlyTime(const std::string& pin_name) const
{
  return EVAL_STA_INST->getRequiredEarlyTime(pin_name);
}

double TimingAPI::getRequiredLateTime(const std::string& pin_name) const
{
  return EVAL_STA_INST->getRequiredLateTime(pin_name);
}

double TimingAPI::reportWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return EVAL_STA_INST->reportWNS(clock_name, mode);
}

double TimingAPI::reportTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return EVAL_STA_INST->reportTNS(clock_name, mode);
}

void TimingAPI::updateTiming(const std::vector<TimingNet*>& timing_net_list, int32_t dbu_unit)
{
  EVAL_STA_INST->updateTiming(timing_net_list, dbu_unit);
}

void TimingAPI::updateTiming(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                             const int& propagation_level, int32_t dbu_unit)
{
  EVAL_STA_INST->updateTiming(timing_net_list, name_list, propagation_level, dbu_unit);
}

bool TimingAPI::isClockNet(const std::string& net_name) const
{
  return EVAL_STA_INST->isClockNet(net_name);
}

}  // namespace ieval