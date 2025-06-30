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
 * @file PwrCalcSwitchPower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc switch power.
 * @version 0.1
 * @date 2023-04-21
 */

#pragma once

#include <fstream>
#include <iostream>

#include "core/PwrAnalysisData.hh"
#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"

namespace ipower {

/**
 * @brief Calc switch power.
 *
 */
class PwrCalcSwitchPower : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  auto& takeSwitchPowers() { return _switch_powers; }

  void printSwitchPower(std::ostream& out, PwrGraph* the_graph);

 private:
  void addSwitchPower(std::unique_ptr<PwrSwitchData> power_data) {
    _switch_powers.emplace_back(std::move(power_data));
  }

  std::vector<std::unique_ptr<PwrSwitchData>>
      _switch_powers;               //!< The switch power.
  double _switch_power_result = 0;  //!< the sum data of switch power.
};
}  // namespace ipower