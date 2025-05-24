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
void TestTiming(const string& db_config_path);

void PrintUsage(const char* program_name)
{
  std::cout << "timing evaluation" << std::endl;
  std::cout << "Usage: " << program_name << " <function_name>" << std::endl;
  std::cout << "Available parameters:" << std::endl;
  std::cout << "  <db_config_path> Path to the database configuration file." << std::endl;
  std::cout << "  --help, -h       Show this help message and exit." << std::endl;
}

int main(const int argc, const char* argv[])
{
  if (argc == 2) {
    if (const std::string arg = argv[1]; arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      TestTiming(arg);
    }
    return 0;
  }
  std::cerr << "Error: Incorrect number of arguments." << std::endl;
  PrintUsage(argv[0]);
  return 1;
}

void TestTiming(const string& db_config_path)
{
  // dmInst->init("/data/project_share/dataset_baseline/gcd/workspace/config/iEDA_config/db_default_config.json");
  // auto config = dmInst->get_config();
  // config.set_output_path("/home/liweiguo/project/iEDA/scripts/design/eval/result");

  // iPLAPIInst.initAPI("/data/project_share/dataset_baseline/gcd/workspace/config/iEDA_config/pl_default_config.json",
  //                    dmInst->get_idb_builder());
  // iPLAPIInst.runFlow();
  dmInst->init(db_config_path);
  auto config = dmInst->get_config();
  config.set_output_path("/home/liweiguo/project/iEDA/scripts/design/eval/result");
  dmInst->readLef(std::vector<std::string>{config.get_tech_lef_path()}, true);
  dmInst->readLef(config.get_lef_paths());
  dmInst->readDef("/data/project_share/dataset_baseline/s15850/workspace/output/iEDA/result/s15850_place.def.gz");

  auto* timing_api = ieval::TimingAPI::getInst();
  // timing_api->runSTA();
  timing_api->evalTiming("FLUTE");
  auto summary = timing_api->evalDesign();
  LOG_INFO << ">> Design Timing Evaluation: ";
  for (auto routing_type : {"HPWL", "FLUTE", "SALT", "EGR", "DR"}) {
    if (summary.contains(routing_type) == false) {
      continue;
    }
    auto timing_summary = summary[routing_type];
    LOG_INFO << "Routing type: " << routing_type;
    for (auto& clock_timing : timing_summary.clock_timings) {
      LOG_INFO << "Clock: " << clock_timing.clock_name << " Setup WNS: " << clock_timing.setup_wns
               << " Setup TNS: " << clock_timing.setup_tns << " Hold WNS: " << clock_timing.hold_wns
               << " Hold TNS: " << clock_timing.hold_tns << " Suggest freq: " << clock_timing.suggest_freq;
    }
    LOG_INFO << "Static power: " << timing_summary.static_power;
    LOG_INFO << "Dynamic power: " << timing_summary.dynamic_power;
  }
}
