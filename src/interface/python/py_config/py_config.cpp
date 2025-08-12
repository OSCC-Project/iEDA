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
#include "py_config.h"

#include <flow.h>
#include <idm.h>
#include <tool_manager.h>

namespace python_interface {

bool flow_init(const std::string& flow_config)
{
  bool init_ok = iplf::plfInst->initFlow(flow_config);
  return init_ok;
}

bool db_init(const std::string& config_path, const std::string& tech_lef_path, const std::vector<std::string>& lef_paths,
             const std::string& def_path, const std::string& verilog_path, const std::string& output_path, const std::string& feature_path,
             const std::vector<std::string>& lib_paths, const std::string& sdc_path)
{
  idm::DataConfig& dm_config = dmInst->get_config();
  if (not config_path.empty()) {
    bool init_ok = dm_config.initConfig(config_path);
    if (not init_ok) {
      return false;
    }
  }
  if (not tech_lef_path.empty()) {
    dm_config.set_tech_lef_path(tech_lef_path);
  }
  if (not lef_paths.empty()) {
    dm_config.set_lef_paths(lef_paths);
  }
  if (not def_path.empty()) {
    dm_config.set_def_path(def_path);
  }
  if (not verilog_path.empty()) {
    dm_config.set_verilog_path(verilog_path);
  }
  if (not output_path.empty()) {
    dm_config.set_output_path(output_path);
  }
  if (not lib_paths.empty()) {
    dm_config.set_lib_paths(lib_paths);
  }
  if (not sdc_path.empty()) {
    dm_config.set_sdc_path(sdc_path);
  }
  if (not feature_path.empty()) {
    dm_config.set_feature_path(feature_path);
  }
  return true;
}

}  // namespace python_interface