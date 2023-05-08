/**
 * @file PwrCalcInternalPower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc internal power.
 * @version 0.1
 * @date 2023-04-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

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
      _internal_powers;  //!< The internal power.
  double _internal_power_result = 0; //!< the sum data of internal power.
};

}  // namespace ipower
