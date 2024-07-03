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
 * @file PwrCalcLeakagePower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc leakage power.
 * @version 0.1
 * @date 2023-04-13
 */

#pragma once

#include <fstream>
#include <iostream>

#include "PwrCalcSPData.hh"
#include "core/PwrAnalysisData.hh"
#include "core/PwrGraph.hh"
#include "include/PwrConfig.hh"
#include "liberty/Lib.hh"

namespace ipower {

/**
 * @brief Calc leakage power.
 *
 */
class PwrCalcLeakagePower : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  auto& takeLeakagePowers() { return _leakage_powers; }

 private:
  double calcLeakagePower(LibLeakagePower* leakage_power, Instance* inst);

  void addLeakagePower(std::unique_ptr<PwrLeakageData> power_data) {
    _leakage_powers.emplace_back(std::move(power_data));
  }

  void printLeakagePower(std::ostream& out);

  std::vector<std::unique_ptr<PwrLeakageData>>
      _leakage_powers;               //!< The leakage power.
  double _leakage_power_result = 0;  //!< the sum data of leakage power.
};
}  // namespace ipower