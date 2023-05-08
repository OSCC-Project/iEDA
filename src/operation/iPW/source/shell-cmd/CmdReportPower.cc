/**
 * @file CmdReportPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief cmd to report power.
 * @version 0.1
 * @date 2023-05-04
 */

#include "PowerShellCmd.hh"
#include "sta/Sta.hh"

namespace ipower {

CmdReportPower::CmdReportPower(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdReportPower::check() { return 1; }

unsigned CmdReportPower::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));

  // set fastest clock for default toggle.
  auto* fastest_clock = ista->getFastestClock();
  PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(),
                             fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = ista->getClocks();

  ipower->setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  // build power graph.
  ipower->buildGraph();

  // build seq graph
  ipower->buildSeqGraph();

  // update power.
  ipower->updatePower();

  // report power.
  // TODO add arg
  std::string output_path = ista->get_design_work_space();
  output_path += Str::printf("/%s.pwr", ista->get_design_name().c_str());

  ipower->reportPower(output_path.c_str(), PwrAnalysisMode::kAveraged);

  return 1;
}

}  // namespace ipower