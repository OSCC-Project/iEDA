/**
 * @file timing_api.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief api for timing & power evaluation
 */

#include "timing_api.hh"

namespace ieval {
TimingAPI::TimingAPI(const std::string& routing_type)
{
  _timing_eval = std::make_unique<TimingEval>(routing_type);
}
}  // namespace ieval