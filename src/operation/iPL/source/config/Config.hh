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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-21 15:22:20
 * @LastEditTime: 2023-03-11 14:37:06
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/config/Config.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_SRC_CONFIG_H
#define IPL_SRC_CONFIG_H

#include <string>

#include "BufferInserterConfig.hh"
#include "DetailPlacerConfig.hh"
#include "FillerConfig.h"
#include "LegalizerConfig.hh"
// #include "MacroPlacerConfig.hh"
#include "NesterovPlaceConfig.hh"
#include "PostGPConfig.hh"
#include "json/json.hpp"

namespace ipl {

// using imp::MacroPlacerConfig;
class Config
{
 public:
  Config() = delete;
  explicit Config(const std::string& json_file);
  ~Config() = default;
  Config(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(const Config&) = delete;
  Config& operator=(Config&&) = delete;

  // function.
  void printConfig();

  // NesterovPlace config.
  NesterovPlaceConfig& get_nes_config() { return _nes_config; }
  // Buffer config.
  BufferInserterConfig& get_buffer_config() { return _buffer_config; }
  // Legalizer config
  LGConfig& get_lg_config() { return _lg_config; }
  // DetailPlacer config
  DPConfig& get_dp_config() { return _dp_config; }
  // Filler config
  FillerConfig& get_filler_config() { return _filler_config; }
  // // MacroPlacer config
  // MacroPlacerConfig& get_mp_config() { return _mp_config; }
  // PostGP config
  PostGPConfig& get_post_gp_config() { return _post_gp_config; }

  int32_t get_ignore_net_degree() const { return _ignore_net_degree; }
  bool isTimingEffort() const { return _is_timing_effort; }
  bool isCongestionEffort() const { return _is_congestion_effort; }

 private:
  // NesterovPlace config.
  NesterovPlaceConfig _nes_config;
  // Buffer config.
  BufferInserterConfig _buffer_config;
  // Legalizer config
  LGConfig _lg_config;
  // DetailPlacer config
  DPConfig _dp_config;
  // Filler config
  FillerConfig _filler_config;
  // // MacroPlacer config
  // MacroPlacerConfig _mp_config;
  PostGPConfig _post_gp_config;
  int32_t _ignore_net_degree;
  bool _is_timing_effort;
  bool _is_congestion_effort;

  void setConfigFromJson(const std::string& json_file);
  void initConfig(const std::string& json_file);
  void initConfigByJson(nlohmann::json json);
  void checkConfig();
  nlohmann::json getDataByJson(nlohmann::json value, std::vector<std::string> flag_list);
};

inline Config::Config(const std::string& json_file)
{
  setConfigFromJson(json_file);
}

}  // namespace ipl

#endif