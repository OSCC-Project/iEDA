/**
 * @file PwrCalcLeakagePower.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc leakage power.
 * @version 0.1
 * @date 2023-04-13
 */

#pragma once

#include "PwrCalcSPData.hh"
#include "core/PwrAnalysisData.hh"
#include "core/PwrGraph.hh"
#include "include/PwrConfig.hh"
#include "liberty/Liberty.hh"

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
  double calcLeakagePower(LibertyLeakagePower* leakage_power, Instance* inst);

  void addLeakagePower(std::unique_ptr<PwrLeakageData> power_data) {
    _leakage_powers.emplace_back(std::move(power_data));
  }

  std::vector<std::unique_ptr<PwrLeakageData>>
      _leakage_powers;               //!< The leakage power.
  double _leakage_power_result = 0;  //!< the sum data of leakage power.
};
}  // namespace ipower