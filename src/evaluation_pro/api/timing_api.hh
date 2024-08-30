/**
 * @file timing_api.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief api for timing & power evaluation
 */

#pragma once
#include "timing_db.hh"
#include "timing_eval.hh"
namespace ieval {

class TimingAPI
{
 public:
  TimingAPI(const std::string& routing_type);

  ~TimingAPI() = default;

 private:
  std::unique_ptr<TimingEval> _timing_eval;
};
}  // namespace ieval
