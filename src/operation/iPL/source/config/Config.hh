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
#include "MacroPlacerConfig.hh"
#include "NesterovPlaceConfig.hh"
#include "json/json.hpp"

namespace ipl {

using imp::MacroPlacerConfig;
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
  // MacroPlacer config
  MacroPlacerConfig& get_mp_config() { return _mp_config; }

  int32_t get_ignore_net_degree() const { return _ignore_net_degree; }
  bool isTimingAwareMode() const { return _is_timing_aware_mode; }

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
  // MacroPlacer config
  MacroPlacerConfig _mp_config;

  int32_t _ignore_net_degree;
  bool _is_timing_aware_mode;

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