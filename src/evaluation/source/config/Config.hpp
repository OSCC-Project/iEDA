#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONFIG_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONFIG_HPP_

#include "CongConfig.hpp"
#include "DBConfig.hpp"
#include "DRCConfig.hpp"
#include "GDSWrapperConfig.hpp"
#include "PowerConfig.hpp"
#include "TimingConfig.hpp"
#include "WLConfig.hpp"
#include "json/json.hpp"

namespace eval {

class Config
{
 public:
  static Config* getOrCreateConfig(const std::string& json_file = "")
  {
    static Config _config(json_file);
    return &_config;
  }
  Config() = delete;
  Config(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(const Config&) = delete;
  Config& operator=(Config&&) = delete;

  DBConfig get_db_config() const { return _db_config; }
  CongConfig get_cong_config() const { return _cong_config; }
  DRCConfig get_drc_config() const { return _drc_config; }
  GDSWrapperConfig get_gds_wrapper_config() const { return _gds_wrapper_config; }
  PowerConfig get_power_config() const { return _power_config; }
  TimingConfig get_timing_config() const { return _timing_config; }
  WLConfig get_wl_config() const { return _wl_config; }

  void printConfig();

 private:
  DBConfig _db_config;
  CongConfig _cong_config;
  DRCConfig _drc_config;
  GDSWrapperConfig _gds_wrapper_config;
  PowerConfig _power_config;
  TimingConfig _timing_config;
  WLConfig _wl_config;

  explicit Config(const std::string& json_file);
  ~Config() = default;

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

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONFIG_HPP_
