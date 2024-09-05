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

void TimingAPI::destroyInst()
{
  ieval::TimingEval::destroyInst();
}

void TimingAPI::initRoutingType(const std::string& routing_type)
{
  ieval::TimingEval::initRoutingType(routing_type);
}
TimingSummary TimingAPI::evalDesign()
{
  return EVAL_STA_INST->evalDesign();
}
double TimingAPI::evalNetPower(const std::string& net_name) const
{
  return EVAL_STA_INST->evalNetPower(net_name);
}
std::map<std::string, double> TimingAPI::evalAllNetPower() const
{
  return EVAL_STA_INST->evalAllNetPower();
}
}  // namespace ieval