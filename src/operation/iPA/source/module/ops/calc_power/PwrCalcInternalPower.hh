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
 * @file PwrCalcInternalPower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc internal power.
 * @version 0.1
 * @date 2023-04-16
 */

#pragma once

#include <fstream>
#include <iostream>

#include "core/PwrAnalysisData.hh"
#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"

namespace ipower {

/**
 * @brief Calc internal power.
 *
 */
class PwrCalcInternalPower : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  auto& takeInternalPowers() { return _internal_powers; }
  void printInternalPower(std::ostream& out, PwrGraph* the_graph);

 private:
  double getToggleData(Pin* pin);
  double calcSPByWhen(const char* when, Instance* inst);

  /*Clac power for pin.*/
  // comb pins
  double calcCombInputPinPower(Instance* inst, Pin* pin,
                               double input_sum_toggle,
                               double output_pin_toggle);

  double calcOutputPinPower(Instance* inst, Pin* pin);

  // seq pins
  double calcClockPinPower(Instance* inst, Pin* pin, double output_pin_toggle);
  double calcSeqInputPinPower(Instance* inst, Pin* pin);

  /*Clac power for instance.*/
  double calcCombInternalPower(Instance* inst);
  double calcSeqInternalPower(Instance* inst);

  void addInternalPower(std::unique_ptr<PwrInternalData> power_data) {
    _internal_powers.emplace_back(std::move(power_data));
  }
  std::vector<std::unique_ptr<PwrInternalData>>
      _internal_powers;               //!< The internal power.
  double _internal_power_result = 0;  //!< the sum data of internal power.
};

}  // namespace ipower
