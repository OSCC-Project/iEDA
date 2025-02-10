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

namespace ista {
enum class AnalysisMode;
}
namespace ieval {
class TimingAPI
{
 public:
  TimingAPI() = default;

  ~TimingAPI() = default;

  static TimingAPI* getInst();

  void runSTA();
  void evalTiming(const std::string& routing_type, const bool& rt_done = false);

  static void destroyInst();

  std::map<std::string, TimingSummary> evalDesign();

  std::map<std::string, std::unordered_map<std::string, double>> evalNetPower() const;

  double getEarlySlack(const std::string& pin_name) const;
  double getLateSlack(const std::string& pin_name) const;
  double getArrivalEarlyTime(const std::string& pin_name) const;
  double getArrivalLateTime(const std::string& pin_name) const;
  double getRequiredEarlyTime(const std::string& pin_name) const;
  double getRequiredLateTime(const std::string& pin_name) const;
  double reportWNS(const char* clock_name, ista::AnalysisMode mode);
  double reportTNS(const char* clock_name, ista::AnalysisMode mode);

  void updateTiming(const std::vector<TimingNet*>& timing_net_list, int32_t dbu_unit);
  void updateTiming(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list, const int& propagation_level,
                    int32_t dbu_unit);

  bool isClockNet(const std::string& net_name) const;

 private:
  static TimingAPI* _timing_api;
};
}  // namespace ieval
