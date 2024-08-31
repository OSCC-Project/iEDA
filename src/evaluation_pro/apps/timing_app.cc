/**
 * @file timing_app.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-29
 * @brief application for timing evaluation
 */

#include "PLAPI.hh"
#include "idm.h"
#include "log/Log.hh"
#include "timing_api.hh"
void TestTiming();

int main()
{
  TestTiming();
  return 0;
}

void TestTiming()
{
  dmInst->init("/data/project_share/dataset_baseline/gcd/workspace/config/iEDA_config/db_default_config.json");
  auto config = dmInst->get_config();
  config.set_output_path("/home/liweiguo/project/iEDA/scripts/design/eval/result");

  iPLAPIInst.initAPI("/data/project_share/dataset_baseline/gcd/workspace/config/iEDA_config/pl_default_config.json",
                     dmInst->get_idb_builder());
  iPLAPIInst.runFlow();
  auto routing_type = "FLUTE";  // "WLM", "HPWL", "FLUTE", "EGR", "DR"
  ieval::TimingAPI timing_api(routing_type);
  auto summary = timing_api.evalDesign();
  LOG_INFO << ">> Design Timing Evaluation: ";
  for (auto& clock_timing : summary.timing) {
    LOG_INFO << "Clock: " << clock_timing.clock_name << " WNS: " << clock_timing.wns << " TNS: " << clock_timing.tns
             << " Suggest freq: " << clock_timing.suggest_freq;
  }
  LOG_INFO << "Static power: " << summary.static_power;
  LOG_INFO << "Dynamic power: " << summary.dynamic_power;
  LOG_INFO << ">> Net Power Evaluation: ";
  LOG_INFO << "Net: clk, Power: " << timing_api.evalNetPower("clk");
}
