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
#include "py_idrc.h"

#include <tool_manager.h>

#include "DRCInterface.hpp"

namespace python_interface {

bool init_drc(const std::string& temp_directory_path, const int& thread_number)
{
  std::map<std::string, std::any> config_map;
  if (temp_directory_path != "") {
    config_map.insert(std::make_pair("-temp_directory_path", temp_directory_path));
  }

  config_map.insert(std::make_pair("-thread_number", thread_number));

  DRCI.initDRC(config_map, false);
  return true;
}

bool run_drc(const std::string& config, const std::string& report)
{
  return iplf::tmInst->autoRunDRC(config, report, true);
}

bool save_drc(const std::string& path)
{
  return iplf::tmInst->saveDrcDetailToFile(path);
}

}  // namespace python_interface