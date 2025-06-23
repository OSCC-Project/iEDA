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
 * @file timing_io.cpp
 * @author qiming chu (me@emin.chat)
 * @brief Timing evaluation IO for tcl interface.
 * @version 0.1
 * @date 2025-06-02
 */

#include "timing_io.h"

#include "json/json.hpp"
#include "log/Log.hh"
#include "timing_api.hh"

namespace ieval {
std::string EvalTiming::_output_path = "./eval_result/timing_result.json";

EvalTiming::EvalTiming() : _routing_type("FLUTE")
{
  setOutputPath("./results");
}

EvalTiming::~EvalTiming() = default;

bool EvalTiming::runTimingEval(const std::string& routing_type)
{
  const auto timing_api = TimingAPI::getInst();
  timing_api->evalTiming(routing_type);
  std::cout << "Timing evaluation completed for routing type: " << routing_type << std::endl;
  return true;
}

void EvalTiming::printTimingResult()
{
  auto* timing_api = TimingAPI::getInst();
  auto summary = timing_api->evalDesign();

  nlohmann::json result_json;

  LOG_INFO << ">> Design Timing Evaluation: ";
  for (const auto routing_type : {"HPWL", "FLUTE", "SALT", "EGR", "DR"}) {
    if (summary.contains(routing_type) == false) {
      continue;
    }

    auto timing_summary = summary[routing_type];
    LOG_INFO << "Routing type: " << routing_type;

    nlohmann::json routing_json;

    nlohmann::json clocks_json = nlohmann::json::array();
    for (const auto& clock_timing : timing_summary.clock_timings) {
      LOG_INFO << "Clock: " << clock_timing.clock_name << " Setup WNS: " << clock_timing.setup_wns
               << " Setup TNS: " << clock_timing.setup_tns << " Hold WNS: " << clock_timing.hold_wns
               << " Hold TNS: " << clock_timing.hold_tns << " Suggest freq: " << clock_timing.suggest_freq;

      nlohmann::json clock_json;
      clock_json["clock_name"] = clock_timing.clock_name;
      clock_json["setup_wns"] = clock_timing.setup_wns;
      clock_json["setup_tns"] = clock_timing.setup_tns;
      clock_json["hold_wns"] = clock_timing.hold_wns;
      clock_json["hold_tns"] = clock_timing.hold_tns;
      clock_json["suggest_freq"] = clock_timing.suggest_freq;

      clocks_json.push_back(clock_json);
    }

    LOG_INFO << "Static power: " << timing_summary.static_power;
    LOG_INFO << "Dynamic power: " << timing_summary.dynamic_power;

    routing_json["clock_timings"] = clocks_json;
    routing_json["static_power"] = timing_summary.static_power;
    routing_json["dynamic_power"] = timing_summary.dynamic_power;

    result_json[routing_type] = routing_json;
  }

  std::string output_path = _output_path;

  if (std::ofstream output_file(output_path); output_file.is_open()) {
    output_file << result_json.dump(2);
    output_file.close();
    LOG_INFO << "Timing evaluation results saved to " << output_path;
  } else {
    LOG_ERROR << "Failed to open file for writing: " << output_path;
  }
}

void EvalTiming::setOutputPath(const std::string& path)
{
  if (path.empty()) {
    std::cout << "Output path is empty, using default path: " << _output_path << std::endl;
    return;
  }
  if (_output_path == path) {
    std::cout << "Output path already exists, using default path: " << _output_path << std::endl;
    return;
  }
  _output_path = path + "/timing_result.json";
  std::cout << "Setting Timing Evaluation report output path to " << _output_path << std::endl;
  std::filesystem::create_directories(std::filesystem::path(path));
}

}  // namespace ieval
