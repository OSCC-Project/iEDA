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
 * @file init_sta.h
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
  WLM,
  HPWL,
  FLUTE,
  EGR,
  DR
}

class InitSTA
{
 public:
  InitSTA(const std::string& work_dir, const RoutingType& routing_type, const unsigned& num_threads = 1, const unsigned& n_worst = 10)
      : _work_dir(work_dir), _num_threads(num_threads), _n_worst(n_worst)
  {
  }
  ~InitSTA();

  void runSTA();

  std::map<std::string, std::map<std::string, double>> getTiming() { return _timing; }
  std::map<std::string, double> getPower() { return _power; }

 private:
  void embeddingSTA();
  void initStaEngine();
  void initPowerEngine();
  void getInfoFromSTA();
  void getInfoFromPW();
  std::string _work_dir = "";
  RoutingType _routing_type = RoutingType::WLM;
  unsigned _num_threads = 1;
  unsigned _n_worst = 10;

  std::map<std::string, std::map<std::string, double>> _timing;
  std::map<std::string, double> _power;
};

}  // namespace ieval