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
#include "Config.hpp"

#include <fstream>

#include "EvalLog.hpp"

namespace eval {

void Config::setConfigFromJson(const std::string& json_file)
{
  initConfig(json_file);
  checkConfig();
  printConfig();
}

void Config::initConfig(const std::string& json_file)
{
  std::ifstream in_stream(json_file);
  if (!in_stream.good()) {
    LOG_FATAL << "Cannot open json file : " << json_file << " for reading";
  }
  nlohmann::json json;
  in_stream >> json;
  initConfigByJson(json);
  in_stream.close();
}

void Config::initConfigByJson(nlohmann::json json)
{
  /******************************get data from json*****************************************/
  // DBWrapper
  bool is_db_wrapper = getDataByJson(json, {"Evaluator", "DBWrapper", "enable_wrapper"});
  std::string source = getDataByJson(json, {"Evaluator", "DBWrapper", "source"});
  std::string separator = getDataByJson(json, {"Evaluator", "DBWrapper", "separator"});
  std::vector<std::string> lef_file_list;
  lef_file_list.push_back(getDataByJson(json, {"Evaluator", "DBWrapper", "tech_lef_file"}));
  for (std::string cell_lef_file : getDataByJson(json, {"Evaluator", "DBWrapper", "cell_lef_files"})) {
    lef_file_list.push_back(cell_lef_file);
  }
  std::string def_file = getDataByJson(json, {"Evaluator", "DBWrapper", "def_file"});

  // Congestion
  bool is_congestion_eval = getDataByJson(json, {"Evaluator", "Congestion", "enable_eval"});
  std::string congestion_eval_type = getDataByJson(json, {"Evaluator", "Congestion", "eval_type"});
  int32_t bin_cnt_x = getDataByJson(json, {"Evaluator", "Congestion", "bin_cnt_x"});
  int32_t bin_cnt_y = getDataByJson(json, {"Evaluator", "Congestion", "bin_cnt_y"});
  int32_t tile_size_x = getDataByJson(json, {"Evaluator", "Congestion", "tile_size_x"});
  int32_t tile_size_y = getDataByJson(json, {"Evaluator", "Congestion", "tile_size_y"});
  std::string congestion_output_dir = getDataByJson(json, {"Evaluator", "Congestion", "output_dir"});
  std::string congestion_output_filename = getDataByJson(json, {"Evaluator", "Congestion", "output_filename"});

  // DRC
  bool is_drc_eval = getDataByJson(json, {"Evaluator", "DRC", "enable_eval"});
  std::string drc_output_dir = getDataByJson(json, {"Evaluator", "DRC", "output_dir"});
  std::string drc_output_filename = getDataByJson(json, {"Evaluator", "DRC", "output_filename"});

  // GDSWrapper
  bool is_gds_wrapper_eval = getDataByJson(json, {"Evaluator", "GDSWrapper", "enable_eval"});
  std::string gds_wrapper_output_dir = getDataByJson(json, {"Evaluator", "GDSWrapper", "output_dir"});
  std::string gds_wrapper_output_filename = getDataByJson(json, {"Evaluator", "GDSWrapper", "output_filename"});

  // Power
  bool is_power_eval = getDataByJson(json, {"Evaluator", "Power", "enable_eval"});
  std::string power_output_dir = getDataByJson(json, {"Evaluator", "Power", "output_dir"});
  std::string power_output_filename = getDataByJson(json, {"Evaluator", "Power", "output_filename"});

  // Timing
  bool is_timing_eval = getDataByJson(json, {"Evaluator", "Timing", "enable_eval"});
  std::string sta_workspace = getDataByJson(json, {"Evaluator", "Timing", "sta_workspace"});
  std::string sdc_file = getDataByJson(json, {"Evaluator", "Timing", "sdc_file"});
  std::vector<std::string> lib_file_list;
  for (std::string lib_file : getDataByJson(json, {"Evaluator", "Timing", "lib_file_list"})) {
    lib_file_list.push_back(lib_file);
  }
  std::string timing_output_dir = getDataByJson(json, {"Evaluator", "Timing", "output_dir"});

  // Wirelength
  bool is_wirelength_eval = getDataByJson(json, {"Evaluator", "Wirelength", "enable_eval"});
  std::string wirelength_eval_type = getDataByJson(json, {"Evaluator", "Wirelength", "eval_type"});
  std::string wirelength_output_dir = getDataByJson(json, {"Evaluator", "Wirelength", "output_dir"});
  std::string wirelength_output_filename = getDataByJson(json, {"Evaluator", "Wirelength", "output_filename"});

  /******************************set data from json*****************************************/
  // DBWrapper
  _db_config.enable_wrapper(is_db_wrapper);
  _db_config.set_source(source);
  _db_config.set_separator(separator);
  _db_config.set_lef_file_list(lef_file_list);
  _db_config.set_def_file(def_file);

  // congestion
  _cong_config.enable_eval(is_congestion_eval);
  _cong_config.set_eval_type(congestion_eval_type);
  _cong_config.set_bin_cnt_x(bin_cnt_x);
  _cong_config.set_bin_cnt_y(bin_cnt_y);
  _cong_config.set_tile_size_x(tile_size_x);
  _cong_config.set_tile_size_y(tile_size_y);
  _cong_config.set_output_dir(congestion_output_dir);
  _cong_config.set_output_filename(congestion_output_filename);

  // drc
  _drc_config.enable_eval(is_drc_eval);
  _drc_config.set_output_dir(drc_output_dir);
  _drc_config.set_output_filename(drc_output_filename);

  // gds writer
  _gds_wrapper_config.enable_eval(is_gds_wrapper_eval);
  _gds_wrapper_config.set_output_dir(gds_wrapper_output_dir);
  _gds_wrapper_config.set_output_filename(gds_wrapper_output_filename);

  // power
  _power_config.enable_eval(is_power_eval);
  _power_config.set_output_dir(power_output_dir);
  _power_config.set_output_filename(power_output_filename);

  // timing
  _timing_config.enable_eval(is_timing_eval);
  _timing_config.set_sta_workspace(sta_workspace);
  _timing_config.set_sdc_file(sdc_file);
  _timing_config.set_lib_file_list(lib_file_list);
  _timing_config.set_output_dir(timing_output_dir);

  // wirelength
  _wl_config.enable_eval(is_wirelength_eval);
  _wl_config.set_eval_type(wirelength_eval_type);
  _wl_config.set_output_dir(wirelength_output_dir);
  _wl_config.set_output_filename(wirelength_output_filename);
}

nlohmann::json Config::getDataByJson(nlohmann::json value, std::vector<std::string> flag_list)
{
  int flag_size = flag_list.size();
  if (flag_size == 0) {
    LOG_FATAL << "Config "
              << "The number of json flag is zero!";
  }
  for (int i = 0; i < flag_size; i++) {
    value = value[flag_list[i]];
  }
  if (!value.is_null()) {
    return value;
  }
  std::string key;
  for (int i = 0; i < flag_size; i++) {
    key += flag_list[i];
    if (i < flag_size - 1) {
      key += ".";
    }
  }
  LOG_FATAL << "Config "
            << "The configuration file key=[ " << key << " ] is null! exit...";
}

void Config::checkConfig()
{
}

void Config::printConfig()
{
  LOG_INFO << "==================================Evaluator==================================";
  LOG_INFO << "DBWrapper";
  LOG_INFO << "         enable_wrapper: " << _db_config.enable_wrapper();
  LOG_INFO << "         source: " << _db_config.get_source();
  LOG_INFO << "         separater: " << _db_config.get_separator();
  LOG_INFO << "         lef_file_list: ";
  std::vector<std::string> lef_file_list = _db_config.get_lef_file_list();
  for (size_t i = 0; i < lef_file_list.size(); i++) {
    LOG_INFO << "               " << lef_file_list[i];
  }
  LOG_INFO << "         def_file: " << _db_config.get_def_file();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "Congestion";
  LOG_INFO << "         enable_eval: " << _cong_config.enable_eval();
  LOG_INFO << "         eval_type: " << _cong_config.get_eval_type();
  LOG_INFO << "         bin_cnt_x: " << _cong_config.get_bin_cnt_x();
  LOG_INFO << "         bin_cnt_y: " << _cong_config.get_bin_cnt_y();
  LOG_INFO << "         tile_size_x: " << _cong_config.get_tile_size_x();
  LOG_INFO << "         tile_size_y: " << _cong_config.get_tile_size_y();
  LOG_INFO << "         output_dir: " << _cong_config.get_output_dir();
  LOG_INFO << "         output_filename: " << _cong_config.get_output_filename();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "DRC";
  LOG_INFO << "         enable_eval: " << _drc_config.enable_eval();
  LOG_INFO << "         output_dir: " << _drc_config.get_output_dir();
  LOG_INFO << "         output_filename: " << _drc_config.get_output_filename();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "GDSWrapper";
  LOG_INFO << "         enable_eval: " << _gds_wrapper_config.enable_eval();
  LOG_INFO << "         output_dir: " << _gds_wrapper_config.get_output_dir();
  LOG_INFO << "         output_filename: " << _gds_wrapper_config.get_output_filename();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "Power";
  LOG_INFO << "         enable_eval: " << _power_config.enable_eval();
  LOG_INFO << "         output_dir: " << _power_config.get_output_dir();
  LOG_INFO << "         output_filename: " << _power_config.get_output_filename();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "Timing";
  LOG_INFO << "         enable_eval: " << _timing_config.enable_eval();
  LOG_INFO << "         sta_workspace: " << _timing_config.get_sta_workspace();
  LOG_INFO << "         sdc_file: " << _timing_config.get_sdc_file();
  LOG_INFO << "         lib_file_list: ";
  std::vector<std::string> lib_file_list = _timing_config.get_lib_file_list();
  for (size_t i = 0; i < lib_file_list.size(); i++) {
    LOG_INFO << "               " << lib_file_list[i];
  }
  LOG_INFO << "         output_dir: " << _timing_config.get_output_dir();
  LOG_INFO << "-----------------------------------------------------------------------------";
  LOG_INFO << "Wirelength";
  LOG_INFO << "         enable_eval: " << _wl_config.enable_eval();
  LOG_INFO << "         eval_type: " << _wl_config.get_eval_type();
  LOG_INFO << "         output_dir: " << _wl_config.get_output_dir();
  LOG_INFO << "         output_filename: " << _wl_config.get_output_filename();
  LOG_INFO << "==================================Evaluator==================================";
}

}  // namespace eval
