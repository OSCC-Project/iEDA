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

#include "api/Power.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

unsigned report_power() {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  // set fastest clock for default toggle
  auto* fastest_clock = ista->getFastestClock();
  ipower::PwrClock pwr_fastest_clock(fastest_clock->get_clock_name(),
                                     fastest_clock->getPeriodNs());
  // get sta clocks
  auto clocks = ista->getClocks();

  ipower->setupClock(std::move(pwr_fastest_clock), std::move(clocks));

  {
    ieda::Stats stats;
    LOG_INFO << "build graph and seq graph start";
    // build power graph
    ipower->buildGraph();

    // build seq graph
    ipower->buildSeqGraph();

    LOG_INFO << "build graph and seq graph end";
    double memory_delta = stats.memoryDelta();
    LOG_INFO << "build graph and seq graph memory usage " << memory_delta
             << "MB";
    double time_delta = stats.elapsedRunTime();
    LOG_INFO << "build graph and seq graph time elapsed " << time_delta << "s";
  }

  {
    ieda::Stats stats;
    LOG_INFO << "power annotate vcd start";
    // std::pair begin_end = {0, 50000000};
    // ipower->readVCD("/home/taosimin/T28/vcd/asic_top.vcd", "u0_asic_top",
    //                 begin_end);
    // annotate toggle sp
    ipower->annotateToggleSP();

    LOG_INFO << "power vcd annotate end";
    double memory_delta = stats.memoryDelta();
    LOG_INFO << "power vcd annotate memory usage " << memory_delta << "MB";
    double time_delta = stats.elapsedRunTime();
    LOG_INFO << "power vcd annotate time elapsed " << time_delta << "s";
  }

  // update power.
  ipower->updatePower();

  {
    // report power.
    ieda::Stats stats;
    LOG_INFO << "power report start";

    // TODO add arg
    std::string output_path = ista->get_design_work_space();
    output_path += Str::printf("/%s.pwr", ista->get_design_name().c_str());

    ipower->reportPower(output_path.c_str(),
                        ipower::PwrAnalysisMode::kAveraged);

    LOG_INFO << "power report end";
    double memory_delta = stats.memoryDelta();
    LOG_INFO << "power report memory usage " << memory_delta << "MB";
    double time_delta = stats.elapsedRunTime();
    LOG_INFO << "power report time elapsed " << time_delta << "s";
  }

  return 1;
}

PYBIND11_MODULE(report_power_cpp, m) {
  m.def("report_power_cpp", &report_power);
}