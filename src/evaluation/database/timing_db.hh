/**
 * @file timing_db.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief db for timing
 */

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ieval {
struct ClockTiming
{
  std::string clock_name;
  double setup_wns;
  double setup_tns;
  double hold_wns;
  double hold_tns;
  double suggest_freq;
};
struct TimingSummary
{
  std::vector<ClockTiming> clock_timings;
  double static_power;
  double dynamic_power;
};

struct TimingPin
{
  std::string pin_name;
  int pin_id;
  bool is_real_pin;
  int32_t x;
  int32_t y;
};

struct TimingNet
{
  std::string net_name;
  std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list;
};

}  // namespace ieval