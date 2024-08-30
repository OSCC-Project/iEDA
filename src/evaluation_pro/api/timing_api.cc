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
TimingAPI::TimingAPI(const std::string& routing_type)
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
}  // namespace ieval