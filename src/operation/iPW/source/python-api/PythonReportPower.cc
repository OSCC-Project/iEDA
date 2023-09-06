/**
 * @file .cc
 * @author shaozheqing (707005020@qq.com)
 * @brief the python api for report power
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PythonPower.hh"
#include "api/Power.hh"
#include "sta/Sta.hh"
#include "tcl/ScriptEngine.hh"

namespace ipower {

unsigned report_power() {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  // set fastest clock for default toggle
  auto* fastest_clock = ista->getFastestClock();
  ipower::PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(),
                                     fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = ista->getClocks();

  std::string output_path = ista->get_design_work_space();
  output_path += Str::printf("/%s.pwr", ista->get_design_name().c_str());

  ipower->setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  ipower->runCompleteFlow(output_path);

  return 1;
}

}  // namespace ipower
