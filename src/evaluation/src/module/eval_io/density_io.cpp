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
 * @file density_io.cpp
 * @author qiming chu (me@emin.chat)
 * @brief Density evaluation IO for tcl interface.
 * @version 0.1
 * @date 2025-06-014
 */

#include "density_io.h"

#include "density_api.h"
#include "json/json.hpp"

namespace ieval {
std::string EvalDensity::_output_path = "./eval_result/density_result.json";

bool EvalDensity::runDensityEvalAndOutput(int grid_size, const std::string& stage)
{
  const int32_t _grid_size = grid_size;

  DensityAPI density_api;
  const auto [cell_map_summary, pin_map_summary, net_map_summary] = density_api.densityMap(stage, _grid_size);

  std::cout << "Macro density: " << cell_map_summary.macro_density << std::endl;
  std::cout << "StdCell density: " << cell_map_summary.stdcell_density << std::endl;
  std::cout << "AllCell density: " << cell_map_summary.allcell_density << std::endl;
  std::cout << "Macro pin density: " << pin_map_summary.macro_pin_density << std::endl;
  std::cout << "StdCell pin density: " << pin_map_summary.stdcell_pin_density << std::endl;
  std::cout << "AllCell pin density: " << pin_map_summary.allcell_pin_density << std::endl;
  std::cout << "Local net density: " << net_map_summary.local_net_density << std::endl;
  std::cout << "Global net density: " << net_map_summary.global_net_density << std::endl;
  std::cout << "All net density: " << net_map_summary.allnet_density << std::endl;

  const auto [local_net_density, global_net_density, allnet_density] = density_api.netDensityMap(stage, _grid_size);
  std::cout << "Local net density: " << local_net_density << std::endl;
  std::cout << "Global net density: " << global_net_density << std::endl;
  std::cout << "All net density: " << allnet_density << std::endl;

  nlohmann::json density_json;

  density_json["cell_density"] = {{"macro", cell_map_summary.macro_density},
                                  {"stdcell", cell_map_summary.stdcell_density},
                                  {"allcell", cell_map_summary.allcell_density}};

  density_json["pin_density"] = {{"macro_pin", pin_map_summary.macro_pin_density},
                                 {"stdcell_pin", pin_map_summary.stdcell_pin_density},
                                 {"allcell_pin", pin_map_summary.allcell_pin_density}};

  density_json["net_density"] = {{"local_net", net_map_summary.local_net_density},
                                 {"global_net", net_map_summary.global_net_density},
                                 {"allnet", net_map_summary.allnet_density}};

  density_json["net_density_map"] = {{"local_net", local_net_density}, {"global_net", global_net_density}, {"allnet", allnet_density}};

  density_json["metadata"]
      = {{"grid_size", grid_size}, {"stage", stage}, {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}};

  try {
    if (std::ofstream outfile(_output_path); outfile.is_open()) {
      outfile << density_json.dump(2);
      outfile.close();
      std::cout << "Density evaluation results saved to " << _output_path << std::endl;
    } else {
      std::cerr << "Failed to open output file: " << _output_path << std::endl;
      return false;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error writing JSON output: " << e.what() << std::endl;
    return false;
  }

  return true;
}

EvalDensity::EvalDensity() : _db_config_path("default_path")
{
  setOutputPath("./results");
}

EvalDensity::~EvalDensity() = default;

void EvalDensity::setOutputPath(const std::string& path)
{
  if (path.empty()) {
    std::cout << "Output path is empty, using default path: " << _output_path << std::endl;
    return;
  }
  if (_output_path == path) {
    std::cout << "Output path already exists, using default path: " << _output_path << std::endl;
    return;
  }
  _output_path = path + "/density_result.json";
  std::cout << "Setting Density Evaluation report output path to " << _output_path << std::endl;
  std::filesystem::create_directories(std::filesystem::path(path));
}
}  // namespace ieval
