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
namespace ieval {
enum class RoutingType
{
  kNone,
  kWLM,
  kHPWL,
  kFLUTE,
  kEGR,
  kDR
};

class InitSTA
{
 public:
  InitSTA(const RoutingType& routing_type) : _routing_type(routing_type) {}
  ~InitSTA();

  void runSTA();

  std::map<std::string, std::map<std::string, double>> getTiming() { return _timing; }
  std::map<std::string, double> getPower() { return _power; }

  double evalNetPower(const std::string& net_name) const;
  std::map<std::string, double> evalAllNetPower() const;

 private:
  void callRT();
  void getInfoFromRT();
  void embeddingSTA();
  void initStaEngine();
  void buildRCTree();
  void initPowerEngine();
  void getInfoFromSTA();
  void getInfoFromPW();
  RoutingType _routing_type = RoutingType::kWLM;

  std::map<std::string, std::map<std::string, double>> _timing;
  std::map<std::string, double> _power;
};

}  // namespace ieval