/**
 * @file PwrCalcSwitchPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc switch power.
 * @version 0.1
 * @date 2023-04-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PwrCalcSwitchPower.hh"

namespace ipower {
using ieda::Stats;

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

  PwrArc* arc;
  /*Calc switch power for power net arc.*/
  FOREACH_PWR_ARC(the_graph, arc) {
    if (arc->isInstArc()) {
      continue;
    }

    auto* net = dynamic_cast<PwrNetArc*>(arc)->get_net();
    auto* driver_obj = net->getDriver();

    auto* the_sta_graph = the_graph->get_sta_graph();
    auto driver_sta_vertex = the_sta_graph->findVertex(driver_obj);

    PwrVertex* driver_pwr_vertex = nullptr;
    if (driver_sta_vertex) {
      driver_pwr_vertex = the_graph->staToPwrVertex(*driver_sta_vertex);
    } else {
      LOG_FATAL << "not found driver sta vertex.";
    }

    // TODO  input port
    if (driver_pwr_vertex->is_input_port()) {
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
    // add power analysis data.
    addSwitchPower(std::make_unique<PwrSwitchData>(net, arc_swich_power));
    VERBOSE_LOG(2) << "net  " << net->get_name()
                   << "  switch power: " << arc_swich_power << "mW";

    _switch_power_result += arc_swich_power;
  }

  LOG_INFO << "calc switch power result " << _switch_power_result << "mw";

  LOG_INFO << "calc switch power end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "calc switch power memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "calc switch power time elapsed " << time_delta << "s";
  return 1;
}

}  // namespace ipower