/**
 * @file timing_api.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief api for timing & power evaluation
 */

#pragma once
#include <unordered_map>

#include "timing_db.hh"
namespace ieval {
class TimingAPI
{
 public:
  TimingAPI() = default;

  ~TimingAPI() = default;

  static TimingAPI* getInst();

  static void destroyInst();

  std::map<std::string, TimingSummary> evalDesign();

  std::map<std::string, std::unordered_map<std::string, double>> evalNetPower() const;

 private:
  static TimingAPI* _timing_api;
};
}  // namespace ieval
