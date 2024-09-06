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
#include <string>
#include <unordered_map>
namespace ieval {

class InitSTA
{
 public:
  InitSTA() = default;
  ~InitSTA();
  static InitSTA* getInst();
  static void destroyInst();
  void runSTA();

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> getTiming() const { return _timing; }
  std::map<std::string, std::map<std::string, double>> getPower() const { return _power; }

  std::map<std::string, std::unordered_map<std::string, double>> getNetPower() const { return _net_power; }

 private:
 void initStaEngine();
  void callRT(const std::string& routing_type);
  void buildRCTree(const std::string& routing_type); void initPowerEngine();

  void updateResult(const std::string& routing_type);

  static InitSTA* _init_sta;

  std::map<std::string, std::map<std::string, std::map<std::string, double>>> _timing;
  std::map<std::string, std::map<std::string, double>> _power;
  std::map<std::string, std::unordered_map<std::string, double>> _net_power;
};

}  // namespace ieval