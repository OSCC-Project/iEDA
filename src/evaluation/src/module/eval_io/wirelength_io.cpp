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
 * @file wirelength_io.cpp
 * @author qiming chu (me@emin.chat)
 * @brief Wirelength evaluation IO for tcl interface.
 * @version 0.1
 * @date 2025-06-14
 */

#include "wirelength_io.h"

#include <filesystem>
#include <fstream>

#include "json/json.hpp"
#include "wirelength_api.h"

namespace ieval {
std::string EvalWirelength::_output_path = "./eval_result/wirelength_result.json";

EvalWirelength::EvalWirelength()
{
  setOutputPath("./result");
}

EvalWirelength::~EvalWirelength() = default;

bool EvalWirelength::runWirelengthEvalAndOutput()
{
  std::cout << "Running wirelength evaluation..." << std::endl;
  WirelengthAPI wirelength_api;
  auto [HPWL, FLUTE, HTree, VTree, GRWL] = wirelength_api.totalWL();
  std::cout << "Total HPWL: " << HPWL << std::endl;
  std::cout << "Total FLUTE: " << FLUTE << std::endl;
  std::cout << "Total HTree: " << HTree << std::endl;
  std::cout << "Total VTree: " << VTree << std::endl;
  std::cout << "Total GRWL: " << GRWL << std::endl;

  nlohmann::json json_output;
  json_output["HPWL"] = HPWL;
  json_output["FLUTE"] = FLUTE;
  json_output["HTree"] = HTree;
  json_output["VTree"] = VTree;

  const std::string output_dir = _output_path.empty() ? "./results/" : _output_path;

  std::string json_path = output_dir;
  std::ofstream json_file(json_path);
  if (json_file.is_open()) {
    json_file << json_output.dump(2);
    json_file.close();
    std::cout << "Wirelength summary saved to: " << json_path << std::endl;
  } else {
    std::cerr << "Failed to open file: " << json_path << std::endl;
  }
  return true;
}

void EvalWirelength::setOutputPath(const std::string& path)
{
  if (path == "") {
    std::cout << "Output path is empty, using default path: " << _output_path << std::endl;
  }
  if (_output_path == path) {
    std::cout << "Output path is already exists, using default path: " << _output_path << std::endl;
    return;
  }
  _output_path = path + "/wirelength_result.json";
  std::cout << "[Evaluate Wirelength] Output path set to: " << _output_path << std::endl;
  std::filesystem::create_directories(std::filesystem::path(path));
}
}  // namespace ieval