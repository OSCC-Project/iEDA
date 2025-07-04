// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrCalcSPData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief clac sp data.
 * @version 0.1
 * @date 2023-04-18
 */
#pragma once

#include "core/PwrGraph.hh"
#include "include/PwrConfig.hh"
#include "liberty/Lib.hh"
#include "netlist/Instance.hh"

namespace ipower {
/**
 * calc sp data
 */
class PwrCalcSPData : public PwrFunc {
 public:
  double getSPData(std::string_view port_name, Instance* inst);
  double calcSPData(RustLibertyExpr* expr, Instance* inst);
};
}  // namespace ipower