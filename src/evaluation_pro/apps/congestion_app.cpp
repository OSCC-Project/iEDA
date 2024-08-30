#include <iostream>

#include "congestion_api.h"

void test_egr_map();
void test_rudy_map();
void test_congestion_report();

int main()
{
  // test_egr_map();
  test_rudy_map();
  // test_congestion_report();
  return 0;
}

void test_egr_map()
{
  ieval::CongestionAPI congestion_api;

  std::string map_path = "/home/yhqiu/benchmark/AiEDA/application/test/iEDA/rt_temp_directory";

  ieval::EGRMapSummary egr_map_summary = congestion_api.egrMap(map_path);
  std::cout << "egr horizontal sum: " << egr_map_summary.horizontal_sum << std::endl;
  std::cout << "egr vertical sum: " << egr_map_summary.vertical_sum << std::endl;
  std::cout << "egr union sum: " << egr_map_summary.union_sum << std::endl;
}

void test_rudy_map()
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

  ieval::RUDYMapSummary rudy_map_summary = congestion_api.rudyMap(congestion_nets, region, grid_size);
  std::cout << "rudy horizontal: " << rudy_map_summary.rudy_horizontal << std::endl;
  std::cout << "rudy vertical: " << rudy_map_summary.rudy_vertical << std::endl;
  std::cout << "rudy union: " << rudy_map_summary.rudy_union << std::endl;
  std::cout << "lut rudy horizontal: " << rudy_map_summary.lutrudy_horizontal << std::endl;
  std::cout << "lut rudy vertical: " << rudy_map_summary.lutrudy_vertical << std::endl;
  std::cout << "lut rudy union: " << rudy_map_summary.lutrudy_union << std::endl;
}

void test_congestion_report()
{
}