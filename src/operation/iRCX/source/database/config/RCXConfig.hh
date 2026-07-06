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
#pragma once

#include <string>
#include <vector>

#include "Types.hh"

namespace ircx {

#define RCX_CONFIG_INST (ircx::RCXConfig::getInst())

inline constexpr unsigned kDefaultThreadCount = 64U;
inline constexpr F64 kDefaultOperatingTemperature = 25.0;

class RCXConfig
{
 public:
  struct CornerConfig
  {
    std::string name;
    std::vector<F64> temperatures{kDefaultOperatingTemperature};
    std::string itf_file;
    std::string captab_file;
  };

  static auto getInst() -> RCXConfig&
  {
    static RCXConfig inst;
    return inst;
  }

  auto init(const std::string& config_file) -> bool;
  auto reset() -> void;
  auto set_initialized(bool initialized) -> void { _initialized = initialized; }

  auto get_initialized() const -> bool { return _initialized; }
  auto get_config_path() const -> const std::string& { return _config_path; }
  auto get_thread_num() const -> unsigned { return _thread_num; }
  auto get_mapping_file() const -> const std::string& { return _mapping_file; }
  auto get_corners() const -> const std::vector<CornerConfig>& { return _corners; }
  auto get_output_dir() const -> const std::string& { return _output_dir; }
  auto get_report_geometry() const -> bool { return _report_geometry; }

  auto parse(const std::string& json_file) -> bool;

  RCXConfig(const RCXConfig& other) = delete;
  RCXConfig(RCXConfig&& other) = delete;
  auto operator=(const RCXConfig& other) -> RCXConfig& = delete;
  auto operator=(RCXConfig&& other) -> RCXConfig& = delete;

 private:
  RCXConfig() = default;
  ~RCXConfig() = default;

  bool _initialized{false};
  std::string _config_path;

  // settings
  unsigned _thread_num{kDefaultThreadCount};

  // read file
  std::string _mapping_file;
  std::vector<CornerConfig> _corners;

  // report spef
  std::string _output_dir;
  bool _report_geometry{false};
};

}  // namespace ircx
