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
#include "py_irt.h"

#include <tcl_util.h>

#include <string>

#include "RTInterface.hpp"
#include "flow_config.h"
namespace python_interface {

bool initConfigMapByJSON(const std::string& config, std::map<std::string, std::any>& config_map);

bool destroyRT()
{
  RTI.destroyRT();
  return true;
}

bool runERT(std::string& config, std::map<std::string, std::string>& config_dict)
{
  std::map<std::string, std::any> config_map;

  bool pass = false;
  pass = !pass ? initConfigMapByJSON(config, config_map) : pass;
  if (!pass) {
    return false;
  }
  RTI.runERT(config_map);
  return true;
}

bool runRT()
{
  RTI.runRT();
  return true;
}

bool initRT(std::string& config, std::map<std::string, std::string>& config_dict)
{
  iplf::flowConfigInst->set_status_stage("iRT - Routing");

  std::map<std::string, std::any> config_map;

  bool pass = false;
  pass = !pass ? initConfigMapByJSON(config, config_map) : pass;
  if (!pass) {
    return false;
  }
  RTI.initRT(config_map);
  return true;
}

}  // namespace python_interface
