/**
 * @file timing_db.h
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief db for timing
 */

#pragma once

#include <map>
#include <string>

namespace ieval {
struct ClockTiming
{
  std::string clock_name;
  double wns;
  double tns;
  double suggest_freq;
};
struct TimingSummary
{
  std::vector<ClockTiming> timing;
  double static_power;
  double dynamic_power;
};
}  // namespace ieval