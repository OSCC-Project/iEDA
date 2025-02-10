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

namespace ito {
bool JsonParser::parse()
{
  std::ifstream reader(_json_file);
  cout << "[ToConfig Info] begin config, path = " << _json_file << endl;
  if (!reader) {
    std::cout << "[JsonParser Error] Failed to read json file '" << _json_file << "'!" << std::endl;
    return false;
  }

  Json* json = new Json();
  reader >> *json;

  jsonToConfig(json);

  reader.close();
  delete json;

  return true;
}

void JsonParser::jsonToConfig(Json* json)
{
  auto file_path = json->at("file_path");

  vector<string> paths;
  auto lef_files = file_path.at("lef_files");
  for (auto iter = lef_files.begin(); iter != lef_files.end(); ++iter) {
    paths.emplace_back(*iter);
  }
  toConfig->set_lef_files(paths);
  toConfig->set_def_file(file_path.at("def_file").get<string>());
  //   toConfig->set_design_work_space(file_path.at("design_work_space").get<string>());
  toConfig->set_design_work_space(dmInst->get_config().get_output_path() + "./to");
  toConfig->set_sdc_file(file_path.at("sdc_file").get<string>());
  paths.clear();

  auto lib_files = file_path.at("lib_files");
  for (auto iter = lib_files.begin(); iter != lib_files.end(); ++iter) {
    paths.emplace_back((*iter));
  }
  toConfig->set_lib_files(paths);
  paths.clear();
  toConfig->set_output_def_file(dmInst->get_config().get_output_path() + "./to/ito_result.def");
  toConfig->set_report_file(dmInst->get_config().get_output_path() + "./to/report.txt");
  toConfig->set_gds_file(dmInst->get_config().get_output_path() + "./to/to.gds");

  toConfig->set_routing_tree(json->at("routing_tree").get<string>());
  toConfig->set_setup_target_slack(json->at("setup_target_slack").get<float>());
  toConfig->set_hold_target_slack(json->at("hold_target_slack").get<float>());
  toConfig->set_max_insert_instance_percent(json->at("max_insert_instance_percent").get<float>());
  toConfig->set_max_core_utilization(json->at("max_core_utilization").get<float>());
  toConfig->set_fix_fanout(json->at("fix_fanout").get<bool>());
  toConfig->set_optimize_drv(json->at("optimize_drv").get<bool>());
  toConfig->set_optimize_hold(json->at("optimize_hold").get<bool>());
  toConfig->set_optimize_setup(json->at("optimize_setup").get<bool>());

  vector<string> buffers;
  auto drv_bufs = json->at("DRV_insert_buffers");
  for (auto iter = drv_bufs.begin(); iter != drv_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  toConfig->set_drv_insert_buffers(buffers);
  if (toConfig->get_drv_insert_buffers().empty()) {
    cout << "[ToConfig Info] DRV insert buffers is Null" << endl;
  }
  buffers.clear();

  auto setup_bufs = json->at("setup_insert_buffers");
  for (auto iter = setup_bufs.begin(); iter != setup_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  toConfig->set_setup_insert_buffers(buffers);
  if (toConfig->get_setup_insert_buffers().empty()) {
    cout << "[ToConfig Info] setup insert buffers is Null" << endl;
  }
  buffers.clear();

  auto hold_bufs = json->at("hold_insert_buffers");
  for (auto iter = hold_bufs.begin(); iter != hold_bufs.end(); ++iter) {
    buffers.emplace_back(*iter);
  }
  toConfig->set_hold_insert_buffers(buffers);
  if (toConfig->get_hold_insert_buffers().empty()) {
    cout << "[ToConfig Info] hold insert buffers is Null" << endl;
  }
  buffers.clear();

  toConfig->set_number_of_decreasing_slack_iter(json->at("number_of_decreasing_slack_iter").get<int>());
  toConfig->set_max_allowed_buffering_fanout(json->at("max_allowed_buffering_fanout").get<int>());
  toConfig->set_min_divide_fanout(json->at("min_divide_fanout").get<int>());
  toConfig->set_optimize_endpoints_percent(json->at("optimize_endpoints_percent").get<float>());
  toConfig->set_drv_optimize_iter_number(json->at("drv_optimize_iter_number").get<int>());

  // Set specific names prefixes
  auto specific_prefix = json->at("specific_prefix");

  toConfig->set_drv_buffer_prefix(specific_prefix.at("drv").at("make_buffer").get<string>());
  toConfig->set_drv_net_prefix(specific_prefix.at("drv").at("make_net").get<string>());

  toConfig->set_hold_buffer_prefix(specific_prefix.at("hold").at("make_buffer").get<string>());
  toConfig->set_hold_net_prefix(specific_prefix.at("hold").at("make_net").get<string>());

  toConfig->set_setup_buffer_prefix(specific_prefix.at("setup").at("make_buffer").get<string>());
  toConfig->set_setup_net_prefix(specific_prefix.at("setup").at("make_net").get<string>());

  cout << "[ToConfig Info] output report file path = " << toConfig->get_report_file() << endl;
  cout << "[ToConfig Info] output gds file path = " << toConfig->get_gds_file() << endl;
  cout << "[ToConfig Info] hold target slack = " << toConfig->get_hold_target_slack() << endl;
  cout << "[ToConfig Info] setup target slack = " << toConfig->get_setup_target_slack() << endl;
  auto outStrings = [&](std::vector<std::string> names) {
    for (auto name : names) {
      cout << name << ", ";
    }
    cout << endl;
  };
  cout << "[ToConfig Info] routing tree = " << json->at("routing_tree").get<string>() << endl;
  cout << "[ToConfig Info] DRV insert buffer = ";
  outStrings(toConfig->get_drv_insert_buffers());
  cout << "[ToConfig Info] hold insert buffer = ";
  outStrings(toConfig->get_hold_insert_buffers());
  cout << "[ToConfig Info] setup insert buffer = ";
  outStrings(toConfig->get_setup_insert_buffers());
}

}  // namespace ito
