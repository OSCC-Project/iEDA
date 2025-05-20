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
 * @file CongestionEval.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <list>
#include <string>
#include <vector>

// #include "EvalAPI.hpp"
#include "iPNPCommon.hh"

namespace ipnp {
class CongestionEval
{
 public:
  CongestionEval() = default;
  ~CongestionEval() = default;

  std::vector<std::vector<double>> get_map_overflow() { return _map_overflow; }
  auto get_cong_rudy_value() { return _net_cong_rudy; }

  void evalRudyRouting();
  void evalEGR(idb::IdbBuilder* idb_builder);

 private:
  std::vector<std::vector<double>> _map_overflow;
  std::vector<float> _net_cong_rudy;
};

}  // namespace ipnp
