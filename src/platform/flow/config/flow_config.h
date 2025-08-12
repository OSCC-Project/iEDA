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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>

#include "Str.hh"
#include "json.hpp"
using json = nlohmann::json;

using std::string;
using std::vector;

namespace iplf {
#define flowConfigInst PLFConfig::getInstance()

struct ToolsConfig
{
  string run_tcl;
};

struct FlowConfig
{
  string run_synthesis;
  string run_floorplan;
  string run_placer;
  string run_cts;
  string run_router;
  string run_pa;
  string run_drc;
  string run_gui;
  string run_to;
};

struct ConfigPath
{
  string idb_path;
  string ifp_path;
  string ipl_path;
  string irt_path;
  string idrc_path;
  string icts_path;
  string ito_path;
  string ipnp_path;
};

struct EnvironmentInfo
{
  string software_version = "V23.03-OS-01";
  string user = "null";
  string system = "null";
};

struct FlowStatus
{
  string stage = "iDB - iEDA Database";
  double memmory = 0;
  double runtime = 0;
};

class PLFConfig
{
 public:
  static PLFConfig* getInstance()
  {
    if (_instance == nullptr) {
      // _mutex.lock();
      // if (_instance == NULL) {
      // _instance = new PLFConfig();
      // }
      // _mutex.unlock();

      _instance = new PLFConfig();
    }

    return _instance;
  }

  /// getter
  string get_path() { return _path; }

  bool is_flow_running(string flag) { return flag == "ON" ? true : false; }

  bool is_run_tcl() { return is_flow_running(_tools_config.run_tcl); }

  bool is_run_synthesis() { return is_flow_running(_flow_config.run_synthesis); }
  bool is_run_floorplan() { return is_flow_running(_flow_config.run_floorplan); }
  bool is_run_placer() { return is_flow_running(_flow_config.run_placer); }
  bool is_run_cts() { return is_flow_running(_flow_config.run_cts); }
  bool is_run_router() { return is_flow_running(_flow_config.run_router); }
  bool is_run_drc() { return is_flow_running(_flow_config.run_drc); }
  bool is_run_gui() { return is_flow_running(_flow_config.run_gui); }
  bool is_run_to() { return is_flow_running(_flow_config.run_to); }

  string get_idb_path() { return _config_path.idb_path; }
  string get_ifp_path() { return _config_path.ifp_path; }
  string get_ipl_path() { return _config_path.ipl_path; }
  string get_icts_path() { return _config_path.icts_path; }
  string get_irt_path() { return _config_path.irt_path; }
  string get_idrc_path() { return _config_path.idrc_path; }
  string get_ito_path() { return _config_path.ito_path; }
  string get_ipnp_path() { return _config_path.ipnp_path; }

  FlowStatus& get_status() { return _status; }
  string get_status_stage() { return _status.stage; }
  double get_status_runtime() { return _status.runtime; }
  double get_status_memmory() { return _status.memmory; }
  string get_status_runtime_string() { return ieda::Str::printf("%f s", _status.runtime); }
  string get_status_memmory_string() { return ieda::Str::printf("%f MB", _status.memmory); }

  EnvironmentInfo& get_env_info() { return _env_info; }
  string& get_env_info_software_version() { return _env_info.software_version; }
  string& get_env_info_user() { return _env_info.user; }
  string& get_env_info_system() { return _env_info.system; }

  /// setting
  void set_status_stage(string value) { _status.stage = value; }
  void set_status_runtime(double value) { _status.runtime = value; }
  void set_status_memmory(double value) { _status.memmory = value; }
  void add_status_runtime(double value) { _status.runtime += value; }

  void set_env_info_version(string value) { _env_info.software_version = value; }
  void set_env_info_user(string value) { _env_info.user = value; }
  void set_env_info_system(string value) { _env_info.system = value; }

  /// opreator
  bool initConfig(string path = "");

 private:
  static PLFConfig* _instance;
  // static std::mutex _mutex;

  string _path = "";
  ToolsConfig _tools_config;
  FlowConfig _flow_config;
  ConfigPath _config_path;
  FlowStatus _status;
  EnvironmentInfo _env_info;

  PLFConfig() = default;
  ~PLFConfig() = default;
  PLFConfig(const PLFConfig& other) = delete;
  PLFConfig(PLFConfig&& other) = delete;
  PLFConfig& operator=(const PLFConfig& other) = delete;
  PLFConfig& operator=(PLFConfig&& other) = delete;
};

}  // namespace iplf