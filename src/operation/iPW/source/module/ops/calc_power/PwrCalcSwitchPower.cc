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
 * @file PwrCalcSwitchPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc switch power.
 * @version 0.1
 * @date 2023-04-21
 */

#include "PwrCalcSwitchPower.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
using ieda::Stats;

/**
 * @brief print switch power sorted by worst.
 *
 * @param out
 */
void PwrCalcSwitchPower::printSwitchPower(std::ostream& out,
                                          PwrGraph* the_graph) {
  std::ranges::sort(_switch_powers, [](auto& left, auto& right) {
    return left->get_switch_power() > right->get_switch_power();
  });

  auto* seq_graph = the_graph->get_pwr_seq_graph();

  out << "switch power :\n";
  for (auto& elem : _switch_powers) {
    auto* design_net = dynamic_cast<Net*>(elem->get_design_obj());
    PwrSeqVertex* seq_vertex;
    auto* driver_obj = design_net->getDriver();
    if (driver_obj->isPin()) {
      auto* own_instance = driver_obj->get_own_instance();
      seq_vertex = seq_graph->getSeqVertex(own_instance);
    } else {
      seq_vertex = seq_graph->getPortSeqVertex(
          the_graph->getPowerVertex(design_net->getDriver()));
    }

    auto* the_sta_graph = the_graph->get_sta_graph();
    auto driver_sta_vertex = the_sta_graph->findVertex(driver_obj);

    PwrVertex* driver_pwr_vertex = nullptr;
    if (driver_sta_vertex) {
      driver_pwr_vertex = the_graph->staToPwrVertex(*driver_sta_vertex);
    } else {
      LOG_FATAL << "not found driver sta vertex.";
    }

    // get Capacitance
    double cap = (*driver_sta_vertex)->getNetLoad();

    // get Toggle
    double toggle = driver_pwr_vertex->getToggleData(std::nullopt);

    out << std::scientific << design_net->get_name() << " (level  "
        << (seq_vertex ? seq_vertex->get_level() : 0) << ")"
        << " toggle " << (toggle * 1e+9) << " cap " << (cap * 1e-12);
    out << std::fixed << " : " << elem->get_switch_power() << " mW"
        << "\n";
  }
}

/**
 * @brief Calc switch power.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrCalcSwitchPower::operator()(PwrGraph* the_graph) {
  Stats stats;
  LOG_INFO << "calc switch power start";
  set_the_pwr_graph(the_graph);

  auto* sta_graph = the_graph->get_sta_graph();
  auto* nl = sta_graph->get_nl();

  Net* net;
  /*Calc switch power for power net arc.*/
  FOREACH_NET(nl, net) {
    if (net->getLoads().empty()) {
      LOG_INFO << "net " << net->get_name()
               << " has no load, skip switch power calculation.";
      continue;
    }

    auto* driver_obj = net->getDriver();
    if (!driver_obj) {
      LOG_INFO << "net " << net->get_name()
               << " has no driver, skip switch power calculation.";
      continue;
    }

    if (driver_obj->isPort() &&
        ((net->getLoads().size() == 1) && net->getLoads().front()->isPort())) {
      continue;
    }

    auto* the_sta_graph = the_graph->get_sta_graph();
    auto driver_sta_vertex = the_sta_graph->findVertex(driver_obj);

    PwrVertex* driver_pwr_vertex = nullptr;
    if (driver_sta_vertex) {
      driver_pwr_vertex = the_graph->staToPwrVertex(*driver_sta_vertex);
    } else {
      // LOG_FATAL << "not found driver sta vertex.";
      LOG_ERROR << "not found driver sta vertex.";
      continue;
    }

    // get VDD
    auto driver_voltage = driver_pwr_vertex->getDriveVoltage();
    if (!driver_voltage) {
      LOG_FATAL << "can not get driver voltage.";
    }
    double vdd = driver_voltage.value();

    // get Capacitance
    double cap = (*driver_sta_vertex)->getNetLoad();

    // get Toggle
    double toggle = driver_pwr_vertex->getToggleData(std::nullopt);

    // calc swich power of the arc.
    // swich_power = k*toggle*Cap*(VDD^2)
    double arc_swich_power = c_switch_power_K * toggle * cap * vdd * vdd;
    auto switch_data =
        std::make_unique<PwrSwitchData>(net, MW_TO_W(arc_swich_power));
    switch_data->set_nom_voltage(vdd);
    // add power analysis data.
    addSwitchPower(std::move(switch_data));
    VERBOSE_LOG(2) << "net  " << net->get_name()
                   << "  switch power: " << arc_swich_power << "mW";

    // for debug
    if (0) {
      std::ofstream out_debug("switch_out.txt");

      out_debug << "toggle :" << toggle;
      out_debug << "\ncap :" << cap;
      out_debug << "\nvdd :" << vdd;

      out_debug.close();
    }

    _switch_power_result += arc_swich_power;
  }

#if 0
  // debug switch power
  std::ofstream out("switch.txt");
  printSwitchPower(out, the_graph);
  out.close();
#endif

  LOG_INFO << "calc switch power result " << _switch_power_result << "mw";

  LOG_INFO << "calc switch power end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "calc switch power memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "calc switch power time elapsed " << time_delta << "s";
  return 1;
}

}  // namespace ipower