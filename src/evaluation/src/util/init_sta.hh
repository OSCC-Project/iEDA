// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file init_sta.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 * @brief evaluation with iSTA
 */

#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ista {
enum class AnalysisMode;
}
namespace salt {
class Pin;
}
namespace ieval {

struct TimingNet;

class InitSTA
{
 public:
  InitSTA() = default;
  ~InitSTA();
  static InitSTA* getInst();
  static void destroyInst();
  void runSTA();
  void evalTiming(const std::string& routing_type, const bool& rt_done = false);

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> getTiming() const { return _timing; }
  std::map<std::string, std::map<std::string, double>> getPower() const { return _power; }

  std::map<std::string, std::unordered_map<std::string, double>> getNetPower() const { return _net_power; }

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
  void leaglization(const std::vector<std::shared_ptr<salt::Pin>>& pins);
  void initStaEngine();
  void callRT(const std::string& routing_type);
  void buildRCTree(const std::string& routing_type);
  void initPowerEngine();

  void updateResult(const std::string& routing_type);

  static InitSTA* _init_sta;

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> _timing;
  std::map<std::string, std::map<std::string, double>> _power;
  std::map<std::string, std::unordered_map<std::string, double>> _net_power;
};

}  // namespace ieval