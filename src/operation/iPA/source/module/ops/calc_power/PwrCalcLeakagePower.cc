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
 * @file PwrCalcLeakagePower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc leakage power.
 * @version 0.1
 * @date 2023-04-13
 */

#include "PwrCalcLeakagePower.hh"

#include "core/PwrAnalysisData.hh"

namespace ipower {
using ieda::Stats;

/**
 * @brief Calc leakage power, include analysis conditions.
 *
 * @param leakage_power
 * @return double
 */
double PwrCalcLeakagePower::calcLeakagePower(LibLeakagePower* leakage_power,
                                             Instance* inst) {
  std::string when = leakage_power->get_when();
  double leakage_power_value = leakage_power->get_value();

  if (!when.empty()) {
    /*Parse conditional statements of the leakage power*/
    RustLibertyExprBuilder expr_builder(when.c_str());
    expr_builder.execute();
    auto* expr = expr_builder.get_result_expr();

    /*Calc sp data.*/
    PwrCalcSPData calc_sp_data;
    calc_sp_data.set_the_pwr_graph(get_the_pwr_graph());
    double sp_value = calc_sp_data.calcSPData(expr, inst);
    leakage_power_value *= sp_value;
  }

  return leakage_power_value;
}

/**
 * @brief print leakage power sorted by worst.
 *
 * @param out
 */
void PwrCalcLeakagePower::printLeakagePower(std::ostream& out) {
  std::ranges::sort(_leakage_powers, [](auto& left, auto& right) {
    return left->get_leakage_power() > right->get_leakage_power();
  });

  out << "leakage power :\n";
  for (auto& elem : _leakage_powers) {
    out << elem->get_design_obj()->get_name() << " : "
        << elem->get_leakage_power() << " nW"
        << "\n";
  }
}

/**
 * @brief Calc leakage power of the power vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrCalcLeakagePower::operator()(PwrGraph* the_graph) {
  Stats stats;
  LOG_INFO << "calc leakage power start";

  set_the_pwr_graph(the_graph);

  PwrCell* cell;
  FOREACH_PWR_CELL(the_graph, cell) {
    auto* design_inst = cell->get_design_inst();
    auto* inst_cell = design_inst->get_inst_cell();

    LibLeakagePower* leakage_power;
    double leakage_power_sum_data = 0;
    FOREACH_LEAKAGE_POWER(inst_cell, leakage_power) {
      double leakage_power_data = calcLeakagePower(leakage_power, design_inst);
      leakage_power_sum_data += leakage_power_data;
    }

    // add power analysis data.
    double nom_voltage = inst_cell->get_owner_lib()->get_nom_voltage();
    auto leakage_data =
        std::make_unique<PwrLeakageData>(design_inst, NW_TO_W(leakage_power_sum_data));
    leakage_data->set_nom_voltage(nom_voltage);

    addLeakagePower(std::move(leakage_data));
    VERBOSE_LOG(2) << "cell  " << design_inst->get_name()
                   << "  leakage power: " << leakage_power_sum_data << " nW";

    _leakage_power_result += leakage_power_sum_data;
  }

  // debug leakage power
  // std::ofstream out("leakage.txt");
  // printLeakagePower(out);
  // out.close();

  LOG_INFO << "calc leakage power result " << NW_TO_MW(_leakage_power_result)
           << "mw";

  LOG_INFO << "calc leakage power end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "calc leakage power memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "calc leakage power time elapsed " << time_delta << "s";

  return 1;
}

}  // namespace ipower
