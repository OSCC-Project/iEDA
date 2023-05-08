/**
 * @file PwrCalcSwitchPower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc switch power.
 * @version 0.1
 * @date 2023-04-21
 */

#pragma once

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

 private:

 void addSwitchPower(std::unique_ptr<PwrSwitchData> power_data) {
    _switch_powers.emplace_back(std::move(power_data));
  }

  std::vector<std::unique_ptr<PwrSwitchData>>
      _switch_powers;               //!< The switch power.
  double _switch_power_result = 0;  //!< the sum data of switch power.
};
}  // namespace ipower