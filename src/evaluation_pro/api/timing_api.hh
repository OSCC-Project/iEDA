/**
 * @file timing_api.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief api for timing & power evaluation
 */

#pragma once
#include "timing_db.hh"
namespace ieval {

class TimingAPI
{
 public:
  TimingAPI(const std::string& routing_type);

  ~TimingAPI() = default;

  TimingSummary evalDesign();

  double evalNetPower(const std::string& net_name) const;
};
}  // namespace ieval
