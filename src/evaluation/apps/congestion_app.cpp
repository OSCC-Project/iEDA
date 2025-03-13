/*
 * @FilePath: congestion_app.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */
#include <iostream>

#include "congestion_api.h"
#include "idm.h"

void TestEgrMap();
void TestRudyMap();
void TestEgrOverflow();
void TestRudyUtilization();
// void TestRudyMapFromIDB();
void TestRudyMapFromIDB(const std::string& file_path);
void TestEgrMapFromIDB();
void TestEgrDataStructure();

int main(int argc, char* argv[])
{
  // TestEgrMap();
  // TestRudyMap();
  // TestEgrOverflow();
  // TestRudyUtilization();

  // if (argc > 1) {
  //   std::string map_path(argv[1]);
  //   std::cout << "map_path: " << map_path << std::endl;
  //   TestRudyMapFromIDB(map_path);
  // }

  // TestEgrMapFromIDB();
  TestEgrDataStructure();
  return 0;
}

void TestEgrDataStructure()
{
  std::string congestion_dir = "/home/yhqiu/net_level_collect/benchmark/large_model_test/rt/rt_temp_directory/early_router";

  ieval::CongestionAPI api;
  std::map<std::string, std::vector<std::vector<int>>> egr_map = api.getEGRMap();

  for (const auto& pair : egr_map) {
    std::cout << "Layer: " << pair.first << std::endl;
    const auto& matrix = pair.second;

    for (size_t i = 0; i < matrix.size() && i < 3; ++i) {
      for (const auto& value : matrix[i]) {
        std::cout << value << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void TestEgrMap()
{
  ieval::CongestionAPI congestion_api;

  std::string map_path = "./rt_temp_directory";
  std::string stage = "place";

  ieval::EGRMapSummary egr_map_summary = congestion_api.egrMap(stage, map_path);
  std::cout << "egr horizontal sum: " << egr_map_summary.horizontal_sum << std::endl;
  std::cout << "egr vertical sum: " << egr_map_summary.vertical_sum << std::endl;
  std::cout << "egr union sum: " << egr_map_summary.union_sum << std::endl;
}

void TestRudyMap()
{
  ieval::CongestionAPI congestion_api;

  ieval::CongestionNets congestion_nets;
  ieval::CongestionNet congestion_net_1;
  ieval::CongestionNet congestion_net_2;

  ieval::CongestionPin congestion_pin_1;
  ieval::CongestionPin congestion_pin_2;
  ieval::CongestionPin congestion_pin_3;
  ieval::CongestionPin congestion_pin_4;

  congestion_pin_1.lx = 1;
  congestion_pin_1.ly = 1;
  congestion_pin_2.lx = 5;
  congestion_pin_2.ly = 7;
  congestion_pin_3.lx = 3;
  congestion_pin_3.ly = 3;
  congestion_pin_4.lx = 7;
  congestion_pin_4.ly = 5;

  congestion_net_1.pins.push_back(congestion_pin_1);
  congestion_net_1.pins.push_back(congestion_pin_2);
  congestion_net_2.pins.push_back(congestion_pin_3);
  congestion_net_2.pins.push_back(congestion_pin_4);

  congestion_nets.push_back(congestion_net_1);
  congestion_nets.push_back(congestion_net_2);

  ieval::CongestionRegion region;
  region.lx = 0;
  region.ly = 0;
  region.ux = 10;
  region.uy = 7;

  int32_t grid_size = 2;
  std::string stage = "place";

  ieval::RUDYMapSummary rudy_map_summary = congestion_api.rudyMap(stage, congestion_nets, region, grid_size);
  std::cout << "rudy horizontal: " << rudy_map_summary.rudy_horizontal << std::endl;
  std::cout << "rudy vertical: " << rudy_map_summary.rudy_vertical << std::endl;
  std::cout << "rudy union: " << rudy_map_summary.rudy_union << std::endl;
  std::cout << "lut rudy horizontal: " << rudy_map_summary.lutrudy_horizontal << std::endl;
  std::cout << "lut rudy vertical: " << rudy_map_summary.lutrudy_vertical << std::endl;
  std::cout << "lut rudy union: " << rudy_map_summary.lutrudy_union << std::endl;
}

void TestEgrOverflow()
{
  ieval::CongestionAPI congestion_api;

  std::string map_path = "/home/yhqiu/benchmark/AiEDA/application/test/iEDA/rt_temp_directory";
  std::string stage = "place";

  ieval::OverflowSummary overflow_summary;
  overflow_summary = congestion_api.egrOverflow(stage, map_path);
  std::cout << "total overflow horizontal: " << overflow_summary.total_overflow_horizontal << std::endl;
  std::cout << "total overflow vertical: " << overflow_summary.total_overflow_vertical << std::endl;
  std::cout << "total overflow union: " << overflow_summary.total_overflow_union << std::endl;
  std::cout << "max overflow horizontal: " << overflow_summary.max_overflow_horizontal << std::endl;
  std::cout << "max overflow vertical: " << overflow_summary.max_overflow_vertical << std::endl;
  std::cout << "max overflow union: " << overflow_summary.max_overflow_union << std::endl;
  std::cout << "weighted average overflow horizontal: " << overflow_summary.weighted_average_overflow_horizontal << std::endl;
  std::cout << "weighted average overflow vertical: " << overflow_summary.weighted_average_overflow_vertical << std::endl;
  std::cout << "weighted average overflow union: " << overflow_summary.weighted_average_overflow_union << std::endl;
}

void TestRudyUtilization()
{
  ieval::CongestionAPI congestion_api;
  std::string stage = "place";

  std::string map_path = "/home/yhqiu/benchmark/AiEDA/third_party/iEDA/bin";
  ieval::UtilizationSummary utilization_summary;
  utilization_summary = congestion_api.rudyUtilization(stage, map_path, false);
  std::cout << "max utilization horizontal: " << utilization_summary.max_utilization_horizontal << std::endl;
  std::cout << "max utilization vertical: " << utilization_summary.max_utilization_vertical << std::endl;
  std::cout << "max utilization union: " << utilization_summary.max_utilization_union << std::endl;
  std::cout << "average utilization horizontal: " << utilization_summary.weighted_average_utilization_horizontal << std::endl;
  std::cout << "average utilization vertical: " << utilization_summary.weighted_average_utilization_vertical << std::endl;
  std::cout << "average utilization union: " << utilization_summary.weighted_average_utilization_union << std::endl;

  utilization_summary = congestion_api.rudyUtilization(stage, map_path, true);
  std::cout << "max utilization horizontal: " << utilization_summary.max_utilization_horizontal << std::endl;
  std::cout << "max utilization vertical: " << utilization_summary.max_utilization_vertical << std::endl;
  std::cout << "max utilization union: " << utilization_summary.max_utilization_union << std::endl;
  std::cout << "average utilization horizontal: " << utilization_summary.weighted_average_utilization_horizontal << std::endl;
  std::cout << "average utilization vertical: " << utilization_summary.weighted_average_utilization_vertical << std::endl;
  std::cout << "average utilization union: " << utilization_summary.weighted_average_utilization_union << std::endl;
}

void TestRudyMapFromIDB(const std::string& file_path)
{
  dmInst->init(file_path);

  std::string stage = "place";

  ieval::CongestionAPI congestion_api;
  ieval::RUDYMapSummary rudy_map_summary = congestion_api.rudyMap(stage);

  std::cout << "rudy horizontal: " << rudy_map_summary.rudy_horizontal << std::endl;
  std::cout << "rudy vertical: " << rudy_map_summary.rudy_vertical << std::endl;
  std::cout << "rudy union: " << rudy_map_summary.rudy_union << std::endl;
  std::cout << "lut rudy horizontal: " << rudy_map_summary.lutrudy_horizontal << std::endl;
  std::cout << "lut rudy vertical: " << rudy_map_summary.lutrudy_vertical << std::endl;
  std::cout << "lut rudy union: " << rudy_map_summary.lutrudy_union << std::endl;

  ieval::UtilizationSummary utilization_summary;
  utilization_summary = congestion_api.rudyUtilization(stage, false);
  std::cout << ">>  RUDY " << std::endl;
  std::cout << "max utilization horizontal: " << utilization_summary.max_utilization_horizontal << std::endl;
  std::cout << "max utilization vertical: " << utilization_summary.max_utilization_vertical << std::endl;
  std::cout << "max utilization union: " << utilization_summary.max_utilization_union << std::endl;
  std::cout << "average utilization horizontal: " << utilization_summary.weighted_average_utilization_horizontal << std::endl;
  std::cout << "average utilization vertical: " << utilization_summary.weighted_average_utilization_vertical << std::endl;
  std::cout << "average utilization union: " << utilization_summary.weighted_average_utilization_union << std::endl;

  utilization_summary = congestion_api.rudyUtilization(stage, true);
  std::cout << ">>  LUTRUDY " << std::endl;
  std::cout << "max utilization horizontal: " << utilization_summary.max_utilization_horizontal << std::endl;
  std::cout << "max utilization vertical: " << utilization_summary.max_utilization_vertical << std::endl;
  std::cout << "max utilization union: " << utilization_summary.max_utilization_union << std::endl;
  std::cout << "average utilization horizontal: " << utilization_summary.weighted_average_utilization_horizontal << std::endl;
  std::cout << "average utilization vertical: " << utilization_summary.weighted_average_utilization_vertical << std::endl;
  std::cout << "average utilization union: " << utilization_summary.weighted_average_utilization_union << std::endl;
}

void TestEgrMapFromIDB()
{
  dmInst->init("/data/yhqiu/benchmark/AiEDA/application/benchmark/28nm/gcd/config/db_default_config_test.json");
  std::string stage = "place";

  ieval::CongestionAPI congestion_api;
  ieval::OverflowSummary overflow_summary;
  ieval::EGRMapSummary egr_map_summary = congestion_api.egrMap(stage);
  overflow_summary = congestion_api.egrOverflow(stage);

  std::cout << "egr horizontal sum: " << egr_map_summary.horizontal_sum << std::endl;
  std::cout << "egr vertical sum: " << egr_map_summary.vertical_sum << std::endl;
  std::cout << "egr union sum: " << egr_map_summary.union_sum << std::endl;

  std::cout << "total overflow horizontal: " << overflow_summary.total_overflow_horizontal << std::endl;
  std::cout << "total overflow vertical: " << overflow_summary.total_overflow_vertical << std::endl;
  std::cout << "total overflow union: " << overflow_summary.total_overflow_union << std::endl;
  std::cout << "max overflow horizontal: " << overflow_summary.max_overflow_horizontal << std::endl;
  std::cout << "max overflow vertical: " << overflow_summary.max_overflow_vertical << std::endl;
  std::cout << "max overflow union: " << overflow_summary.max_overflow_union << std::endl;
  std::cout << "weighted average overflow horizontal: " << overflow_summary.weighted_average_overflow_horizontal << std::endl;
  std::cout << "weighted average overflow vertical: " << overflow_summary.weighted_average_overflow_vertical << std::endl;
  std::cout << "weighted average overflow union: " << overflow_summary.weighted_average_overflow_union << std::endl;
}