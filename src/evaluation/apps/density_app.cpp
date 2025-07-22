/*
 * @FilePath: density_app.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include <iostream>

#include "density_api.h"
#include "idm.h"

void TestDensityMap();
void TestDensityMapFromIDB(const string& db_config_path);
void TestMarginMap();

void PrintUsage(const char* program_name) {
  std::cout << "Density Evaluation" << std::endl;
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
      // Here are some test functions that can be uncommented to run
      // TestDensityMap();
      // TestMarginMap();
      TestDensityMapFromIDB(arg);
      return 0;
    }
  }
  std::cerr << "Error: Incorrect number of arguments." << std::endl;
  PrintUsage(argv[0]);
  return 1;
}

void TestDensityMap()
{
  std::string stage = "place";

  ieval::DensityAPI density_api;

  ieval::DensityRegion region;
  region.lx = 0;
  region.ly = 0;
  region.ux = 10;
  region.uy = 7;

  ieval::DensityCells cells;
  ieval::DensityCell cell1;
  cell1.type = "macro";
  cell1.lx = 1;
  cell1.ly = 1;
  cell1.width = 4;
  cell1.height = 6;
  ieval::DensityCell cell2;
  cell2.type = "stdcell";
  cell2.lx = 7;
  cell2.ly = 3;
  cell2.width = 2;
  cell2.height = 2;
  cells.push_back(cell1);
  cells.push_back(cell2);

  int32_t grid_size = 2;
  bool neighbor = true;

  ieval::DensityPins pins;
  ieval::DensityPin pin1;
  pin1.type = "macro";
  pin1.lx = 1;
  pin1.ly = 1;
  ieval::DensityPin pin2;
  pin2.type = "stdcell";
  pin2.lx = 3;
  pin2.ly = 3;
  pins.push_back(pin1);
  pins.push_back(pin2);

  ieval::DensityNets nets;
  ieval::DensityNet net1;
  net1.lx = 0;
  net1.ly = 0;
  net1.ux = 1;
  net1.uy = 1;
  ieval::DensityNet net2;
  net2.lx = 1;
  net2.ly = 1;
  net2.ux = 5;
  net2.uy = 7;
  nets.push_back(net1);
  nets.push_back(net2);

  ieval::CellMapSummary cell_map_summary = density_api.cellDensityMap(cells, region, grid_size, stage);
  std::cout << "Macro density: " << cell_map_summary.macro_density << std::endl;
  std::cout << "StdCell density: " << cell_map_summary.stdcell_density << std::endl;
  std::cout << "AllCell density: " << cell_map_summary.allcell_density << std::endl;

  ieval::PinMapSummary pin_map_summary = density_api.pinDensityMap(pins, region, grid_size, stage, neighbor);
  std::cout << "Macro pin density: " << pin_map_summary.macro_pin_density << std::endl;
  std::cout << "StdCell pin density: " << pin_map_summary.stdcell_pin_density << std::endl;
  std::cout << "AllCell pin density: " << pin_map_summary.allcell_pin_density << std::endl;

  ieval::NetMapSummary net_map_summary = density_api.netDensityMap(nets, region, grid_size, stage, neighbor);
  std::cout << "Local net density: " << net_map_summary.local_net_density << std::endl;
  std::cout << "Global net density: " << net_map_summary.global_net_density << std::endl;
  std::cout << "All net density: " << net_map_summary.allnet_density << std::endl;
}

void TestDensityMapFromIDB(const string& db_config_path)
{
  dmInst->init(db_config_path);
  int32_t grid_size = 2000;
  std::string stage = "place";

  ieval::DensityAPI density_api;
  ieval::DensityMapSummary density_map_summary = density_api.densityMap(stage, grid_size);

  std::cout << "Macro density: " << density_map_summary.cell_map_summary.macro_density << std::endl;
  std::cout << "StdCell density: " << density_map_summary.cell_map_summary.stdcell_density << std::endl;
  std::cout << "AllCell density: " << density_map_summary.cell_map_summary.allcell_density << std::endl;
  std::cout << "Macro pin density: " << density_map_summary.pin_map_summary.macro_pin_density << std::endl;
  std::cout << "StdCell pin density: " << density_map_summary.pin_map_summary.stdcell_pin_density << std::endl;
  std::cout << "AllCell pin density: " << density_map_summary.pin_map_summary.allcell_pin_density << std::endl;
  std::cout << "Local net density: " << density_map_summary.net_map_summary.local_net_density << std::endl;
  std::cout << "Global net density: " << density_map_summary.net_map_summary.global_net_density << std::endl;
  std::cout << "All net density: " << density_map_summary.net_map_summary.allnet_density << std::endl;

  ieval::NetMapSummary net_map = density_api.netDensityMap(stage, grid_size);
  std::cout << "Local net density: " << net_map.local_net_density << std::endl;
  std::cout << "Global net density: " << net_map.global_net_density << std::endl;
  std::cout << "All net density: " << net_map.allnet_density << std::endl;
}

void TestMarginMap()
{
  ieval::DensityAPI density_api;

  ieval::DensityRegion die;
  die.lx = 0;
  die.ly = 0;
  die.ux = 250;
  die.uy = 150;

  ieval::DensityRegion core;
  core.lx = 25;
  core.ly = 25;
  core.ux = 175;
  core.uy = 125;

  ieval::DensityCells cells;
  ieval::DensityCell cell1;
  cell1.type = "macro";
  cell1.lx = 40;
  cell1.ly = 50;
  cell1.width = 90;
  cell1.height = 60;
  ieval::DensityCell cell2;
  cell2.type = "macro";
  cell2.lx = 140;
  cell2.ly = 30;
  cell2.width = 20;
  cell2.height = 40;
  cells.push_back(cell1);
  cells.push_back(cell2);

  int32_t grid_size = 25;

  ieval::MacroMarginSummary macro_margin_summary = density_api.macroMarginMap(cells, die, core, grid_size);
  std::cout << "Horizontal margin: " << macro_margin_summary.horizontal_margin << std::endl;
  std::cout << "Vertical margin: " << macro_margin_summary.vertical_margin << std::endl;
  std::cout << "Union margin: " << macro_margin_summary.union_margin << std::endl;
}
