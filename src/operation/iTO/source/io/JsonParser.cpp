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
#include "JsonParser.h"

namespace ito {
JsonParser *JsonParser::get_json_parser() {
  static JsonParser *parser;
  return parser;
}

void JsonParser::parse(const string &json_file, ToConfig *config) const {
  std::ifstream ifs(json_file);
  if (!ifs) {
    std::cout << "[JsonParser Error] Failed to read json file '" << json_file << "'!"
              << std::endl;
    assert(0);
  } else {
    std::cout << "[iTO info] Read config success : '" << json_file << "'!" << std::endl;
  }
  Json *json = new Json();
  ifs >> *json;

  jsonToConfig(json, config);

  // printConfig(config);

  ifs.close();
  delete json;
}

void JsonParser::jsonToConfig(Json *json, ToConfig *config) const {
  auto file_path = json->at("file_path");

  vector<string> paths;
  auto           lef_files = file_path.at("lef_files");
  for (auto iter = lef_files.begin(); iter != lef_files.end(); ++iter) {
    paths.emplace_back(*iter);
  }
  config->set_lef_files(paths);
  config->set_def_file(file_path.at("def_file").get<string>());
  config->set_design_work_space(file_path.at("design_work_space").get<string>());
  config->set_sdc_file(file_path.at("sdc_file").get<string>());
  paths.clear();

  auto lib_files = file_path.at("lib_files");
  for (auto iter = lib_files.begin(); iter != lib_files.end(); ++iter) {
    paths.emplace_back((*iter));
  }
  config->set_lib_files(paths);
  paths.clear();
  config->set_output_def_file(file_path.at("output_def").get<string>());
  config->set_report_file(file_path.at("report_file").get<string>());
  config->set_gds_file(file_path.at("gds_file").get<string>());

  config->set_setup_target_slack(json->at("setup_slack_margin").get<float>());
  config->set_hold_slack_margin(json->at("hold_slack_margin").get<float>());
  config->set_max_buffer_percent(json->at("max_buffer_percent").get<float>());
  config->set_max_utilization(json->at("max_utilization").get<float>());
  config->set_fix_fanout(json->at("fix_fanout").get<bool>());
  config->set_optimize_drv(json->at("optimize_drv").get<bool>());
  config->set_optimize_hold(json->at("optimize_hold").get<bool>());
  config->set_optimize_setup(json->at("optimize_setup").get<bool>());

  vector<string> buffers;
  auto           drv_bufs = json->at("DRV_insert_buffers");
  for (auto iter = drv_bufs.begin(); iter != drv_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  config->set_drv_insert_buffers(buffers);
  if (config->get_drv_insert_buffers().empty()) {
    cout << "[ToConfig Info] DRV_insert_buffers is Null" << endl;
  }
  buffers.clear();

  auto setup_bufs = json->at("setup_insert_buffers");
  for (auto iter = setup_bufs.begin(); iter != setup_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  config->set_setup_insert_buffers(buffers);
  if (config->get_setup_insert_buffers().empty()) {
    cout << "[ToConfig Info] setup_insert_buffers is Null" << endl;
  }
  buffers.clear();

  auto hold_bufs = json->at("hold_insert_buffers");
  for (auto iter = hold_bufs.begin(); iter != hold_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  config->set_hold_insert_buffers(buffers);
  if (config->get_hold_insert_buffers().empty()) {
    cout << "[ToConfig Info] hold_insert_buffers is Null" << endl;
  }
  buffers.clear();

  config->set_number_passes_allowed_decreasing_slack(
      json->at("number_passes_allowed_decreasing_slack").get<int>());
  config->set_rebuffer_max_fanout(json->at("rebuffer_max_fanout").get<int>());
  config->set_split_load_min_fanout(json->at("split_load_min_fanout").get<int>());

  cout << "[ToConfig Info] hold_slack_margin:\n\t\t" << config->get_hold_target_slack()
       << endl;
}

void JsonParser::printConfig(ToConfig *config) const {
  cout << "[ToConfig Info] def_path:\n\t\t" << config->get_def_file() << endl;
  // std::vector<std::string> lef_paths = config->get_lef_files();
  // for (size_t i = 0; i < lef_paths.size(); i++) {
  //   std::cout << "[Configurator Info]     " << lef_paths[i] << std::endl;
  // }
}
} // namespace ito
