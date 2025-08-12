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
/**
 * @File Name: data_config.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

using std::string;
using std::vector;

namespace idm {
class DataConfig
{
  struct LayerSettings
  {
    string routing_layer_1st;
  };

 public:
  DataConfig() {}
  ~DataConfig() = default;

  /// getter
  string get_config_path() { return _config_path; }
  string get_tech_lef_path() { return _tech_lef_path; }
  vector<string> get_lef_paths()
  {
    vector<string> lef_paths;
    lef_paths.emplace_back(_tech_lef_path);
    for (auto path : _lef_paths) {
      if (path != _tech_lef_path) {
        lef_paths.emplace_back(path);
      }
    }

    return lef_paths;
  }
  string& get_def_path() { return _def_path; }
  string& get_verilog_path() { return _verilog_path; }
  string& get_output_path() { return _output_path; }
  string get_output_path_for_idb() { return _output_path + "/idb.def"; }
  vector<string>& get_lib_paths() { return _lib_paths; }
  string& get_sdc_path() { return _sdc_path; }
  string& get_spef_path() { return _spef_path; }
  string& get_feature_path() { return _feature_path;}

  /// settings
  string& get_routing_layer_1st() { return _settings.routing_layer_1st; }

  /// setter
  void set_tech_lef_path(const string lef_path)
  {
    _tech_lef_path = lef_path;
    std::cout << "[Data config set] tech lef = " << _tech_lef_path << std::endl;
  }
  void set_lef_paths(const vector<string> lef_paths)
  {
    _lef_paths = lef_paths;
    for (auto lef : _lef_paths) {
      std::cout << "[Data config set] lef = " << lef << std::endl;
    }
  }
  void set_def_path(const string def_path)
  {
    _def_path = def_path;
    std::cout << "[Data config set] def = " << _def_path << std::endl;
  }
  void set_verilog_path(const string verilog_path)
  {
    _verilog_path = verilog_path;
    std::cout << "[Data config set] verilog = " << _verilog_path << std::endl;
  }
  void set_output_path(const string output_path)
  {
    _output_path = output_path;
    std::cout << "[Data config set] output dir = " << _output_path << std::endl;
  }
  void set_lib_paths(const vector<string> lib_paths)
  {
    _lib_paths = lib_paths;
    for (auto lib : _lib_paths) {
      std::cout << "[Data config set] lib = " << lib << std::endl;
    }
  }
  void set_sdc_path(const string sdc_path)
  {
    _sdc_path = sdc_path;
    std::cout << "[Data config set] sdc = " << _sdc_path << std::endl;
  }

  void set_spef_path(const string spef_path)
  {
    _spef_path = spef_path;
    std::cout << "[Data config set] spef = " << _spef_path << std::endl;
  }

  void set_feature_path(const string feature_path)
  {
    _feature_path = feature_path;
    std::cout << "[Data config set] feature = " << _feature_path << std::endl;
  }

  void set_routing_layer_1st(string layer) { _settings.routing_layer_1st = layer; }

  /// function
  bool initConfig(string config_path);
  bool checkAllFile();

  /// check file exist
  bool checkFilePath(string path);

 private:
  string _config_path;
  string _tech_lef_path;
  vector<string> _lef_paths;
  string _def_path;
  string _verilog_path;
  string _output_path;
  vector<string> _lib_paths;
  string _sdc_path;
  string _spef_path;
  string _feature_path;

  LayerSettings _settings;
};

}  // namespace idm
