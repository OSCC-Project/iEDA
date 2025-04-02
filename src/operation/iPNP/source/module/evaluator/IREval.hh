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
 * @file IREval.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "iPNPCommon.hh"

// 前向声明
namespace ipower {
class Instance;
}

namespace ipnp {
class IREval
{
 public:
  IREval() = default;
  ~IREval() = default;

  double get_ir_value() { return ir_value; }
  std::vector<std::vector<double>> get_ir_map() { return ir_map; }
  
  // 运行IR分析
  unsigned runIRAnalysis(const std::string& power_net_name = "VDD");

 private:
  // 处理IR分析结果
  void processIRResults(const std::map<std::pair<double, double>, double>& ir_drop_map);

  std::vector<std::vector<double>> ir_map;
  double ir_value{0.0};
};

}  // namespace ipnp
