// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "RCXConfig.hh"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "PathUtils.hh"
#include "json/json.hpp"
#include "log/Log.hh"

namespace ircx {

namespace fs = std::filesystem;

namespace {

auto parseCornerConfig(const nlohmann::json& corner_json,
                       const fs::path& config_dir,
                       std::string_view field_name,
                       RCXConfig::CornerConfig& corner_config) -> bool
{
  if (!corner_json.is_object()) {
    LOG_ERROR << "invalid corner config: " << field_name << ", expected object";
    return false;
  }

  const std::string field_name_str(field_name);
  const std::string name_field = field_name_str + ".name";
  const std::string temperature_field = field_name_str + ".temperature";
  const std::string itf_field = field_name_str + ".itf_file";
  const std::string captab_field = field_name_str + ".captab_file";

  // name
  if (!corner_json.contains("name")) {
    LOG_ERROR << "missing required corner field: " << name_field
              << ", expected string";
    return false;
  }
  if (!corner_json["name"].is_string()) {
    LOG_ERROR << "invalid required corner field: " << name_field
              << ", expected string";
    return false;
  }

  // temperature
  if (corner_json.contains("temperature")) {
    if (!corner_json["temperature"].is_array()) {
      LOG_ERROR << "invalid optional corner field: " << temperature_field
                << ", expected array of numbers";
      return false;
    }
    if (corner_json["temperature"].empty()) {
      LOG_ERROR << "invalid optional corner field: " << temperature_field
                << ", must not be empty";
      return false;
    }

    corner_config.temperatures.clear();
    corner_config.temperatures.reserve(corner_json["temperature"].size());
    for (size_t idx = 0; idx < corner_json["temperature"].size(); ++idx) {
      const auto& temperature_json = corner_json["temperature"][idx];
      const std::string temperature_item_field = temperature_field + "["
                                                 + std::to_string(idx) + "]";
      if (!temperature_json.is_number()) {
        LOG_ERROR << "invalid optional corner field: " << temperature_item_field
                  << ", expected number";
        return false;
      }

      const F64 temperature = temperature_json.get<F64>();
      if (!std::isfinite(temperature)) {
        LOG_ERROR << "invalid optional corner field: " << temperature_item_field
                  << ", expected finite number";
        return false;
      }
      corner_config.temperatures.push_back(temperature);
    }
  } else {
    LOG_WARNING << "missing optional corner field: " << temperature_field
                << ", use default operating temperature list: "
                << kDefaultOperatingTemperature;
  }

  // itf_file
  if (!corner_json.contains("itf_file")) {
    LOG_ERROR << "missing required corner field: " << itf_field
              << ", expected string";
    return false;
  }
  if (!corner_json["itf_file"].is_string()) {
    LOG_ERROR << "invalid required corner field: " << itf_field
              << ", expected string";
    return false;
  }

  // captab_file
  if (!corner_json.contains("captab_file")) {
    LOG_ERROR << "missing required corner field: " << captab_field
              << ", expected string";
    return false;
  }
  if (!corner_json["captab_file"].is_string()) {
    LOG_ERROR << "invalid required corner field: " << captab_field
              << ", expected string";
    return false;
  }

  corner_config.name = string::trim(corner_json["name"].get<std::string>());
  corner_config.itf_file = path::resolve(config_dir, corner_json["itf_file"].get<std::string>());
  corner_config.captab_file = path::resolve(config_dir, corner_json["captab_file"].get<std::string>());

  bool valid = true;
  if (corner_config.name.empty()) {
    LOG_ERROR << "invalid required corner field: " << name_field
              << ", must not be empty";
    valid = false;
  }
  if (corner_config.itf_file.empty()) {
    LOG_ERROR << "invalid required corner field: " << itf_field
              << ", must not be empty";
    valid = false;
  }
  if (corner_config.captab_file.empty()) {
    LOG_ERROR << "invalid required corner field: " << captab_field
              << ", must not be empty";
    valid = false;
  }

  return valid;
}

}  // namespace

auto RCXConfig::init(const std::string& config_file) -> bool
{
  reset();
  return parse(config_file);
}

auto RCXConfig::reset() -> void
{
  _initialized = false;
  _config_path.clear();
  _thread_num = kDefaultThreadCount;
  _mapping_file.clear();
  _corners.clear();
  _output_dir.clear();
  _report_geometry = false;
}

auto RCXConfig::parse(const std::string& json_file) -> bool
{
  if (json_file.empty()) {
    LOG_ERROR << "config path is empty";
    return false;
  }

  std::ifstream config_stream(json_file);
  if (!config_stream.is_open()) {
    LOG_ERROR << "failed to open config file: " << json_file;
    return false;
  }

  try {
    nlohmann::json json;
    config_stream >> json;

    const fs::path absolute_config_path = path::absolute(json_file);
    const fs::path config_dir = absolute_config_path.parent_path();

    // thread_num
    if (!json.contains("thread_num")) {
      LOG_ERROR << "missing required config field: thread_num, expected integer";
      return false;
    }
    if (!json["thread_num"].is_number_integer()) {
      LOG_ERROR << "invalid required config field: thread_num, expected integer";
      return false;
    }
    const int thread_num = json["thread_num"].get<int>();
    _thread_num = thread_num <= 0 ? 1U : static_cast<unsigned>(thread_num);


    // output
    if (json.contains("output")) {
      if (!json["output"].is_string()) {
        LOG_ERROR << "invalid optional config field: output, expected string";
        return false;
      }
      _output_dir = path::resolve(config_dir, json["output"].get<std::string>());
      if (_output_dir.empty()) {
        LOG_WARNING << "empty optional config field: output, use default output directory: .";
        _output_dir = ".";
      }
    } else {
      LOG_WARNING << "missing optional config field: output, use default output directory: .";
      _output_dir = ".";
    }

    // report_geometry for spef
    if (json.contains("report_geometry")) {
      if (!json["report_geometry"].is_boolean()) {
        LOG_ERROR << "invalid optional config field: report_geometry, expected boolean";
        return false;
      }
      _report_geometry = json["report_geometry"].get<bool>();
    }

    // mapping_file
    if (!json.contains("mapping_file")) {
      LOG_ERROR << "missing required config field: mapping_file, expected string";
      return false;
    }
    if (!json["mapping_file"].is_string()) {
      LOG_ERROR << "invalid required config field: mapping_file, expected string";
      return false;
    }
    _mapping_file = path::resolve(config_dir, json["mapping_file"].get<std::string>());

    // corners
    if (!json.contains("corners")) {
      LOG_ERROR << "missing required config field: corners, expected object or array";
      return false;
    }

    const auto& corners_json = json["corners"];
    bool valid = true;

    if (!corners_json.is_array() && !corners_json.is_object()) {
      LOG_ERROR << "invalid required config field: corners, expected object or array";
      return false;
    }
    if (corners_json.is_array() && corners_json.empty()) {
      LOG_ERROR << "invalid required config field: corners, must not be empty";
      return false;
    }

    const bool corners_is_array = corners_json.is_array();
    const size_t corner_count = corners_is_array ? corners_json.size() : 1;
    _corners.reserve(corner_count);
    for (size_t idx = 0; idx < corner_count; ++idx) {
      const auto& corner_json = corners_is_array ? corners_json[idx] : corners_json;
      const std::string corner_field = "corners[" + std::to_string(idx) + "]";

      CornerConfig corner_config;
      valid &= parseCornerConfig(corner_json, config_dir, corner_field, corner_config);
      _corners.emplace_back(std::move(corner_config));
    }

    _config_path = absolute_config_path.string();

    if (_mapping_file.empty()) {
      LOG_ERROR << "invalid required config field: mapping_file, must not be empty";
      valid = false;
    }

    return valid;
  } catch (const std::exception& e) {
    LOG_ERROR << "failed to parse config file: " << json_file << ": " << e.what();
    return false;
  }
}

}  // namespace ircx
