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

#include "idm.h"

namespace ino {
JsonParser *JsonParser::get_json_parser() {
  static JsonParser *parser;
  return parser;
}

void JsonParser::parse(const string &json_file, NoConfig *config) const {
  std::ifstream ifs(json_file);
  if (!ifs) {
    std::cout << "[JsonParser Error] Failed to read json file '" << json_file << "'!"
              << std::endl;
    assert(0);
  }
  Json *json = new Json();
  ifs >> *json;

  jsonToConfig(json, config);

  // printConfig(config);

  ifs.close();
  delete json;
}

void JsonParser::jsonToConfig(Json *json, NoConfig *config) const {
  auto file_path = json->at("file_path");

  vector<string> paths;
  auto           lef_files = file_path.at("lef_files");
  for (auto iter = lef_files.begin(); iter != lef_files.end(); ++iter) {
    paths.emplace_back(*iter);
  }
  config->set_lef_files(paths);
  config->set_def_file(file_path.at("def_file").get<string>());
  config->set_design_work_space(dmInst->get_config().get_output_path() + "./no");
  config->set_sdc_file(file_path.at("sdc_file").get<string>());
  paths.clear();

  auto lib_files = file_path.at("lib_files");
  for (auto iter = lib_files.begin(); iter != lib_files.end(); ++iter) {
    paths.emplace_back((*iter));
  }
  config->set_lib_files(paths);
  paths.clear();
  config->set_output_def_file(dmInst->get_config().get_output_path() +
                              "./no/ino_result.def");
  config->set_report_file(dmInst->get_config().get_output_path() + "./no/report.txt");

  config->set_insert_buffer(json->at("insert_buffer").get<string>());
  if (config->get_insert_buffer().empty()) {
    cout << "[Config Info] insert_buffer is Null" << endl;
  }

  config->set_max_fanout(json->at("max_fanout").get<int>());
}

void JsonParser::printConfig(NoConfig *config) const {
  cout << "[Config Info] def_path:\n\t\t" << config->get_def_file() << endl;
  // std::vector<std::string> lef_paths = config->get_lef_files();
  // for (size_t i = 0; i < lef_paths.size(); i++) {
  //   std::cout << "[Configurator Info]     " << lef_paths[i] << std::endl;
  // }
}
} // namespace ino
