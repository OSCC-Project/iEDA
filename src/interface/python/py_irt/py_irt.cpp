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

#include <iRT/api/RTAPI.hpp>
#include <string>
namespace python_interface {

bool destroyRT()
{
  RTAPIInst.destroyRT();
  return true;
}

bool runDR()
{
  RTAPIInst.runDR();
  return true;
}

bool runGR()
{
  RTAPIInst.runGR();
  return true;
}

bool runRT()
{
  RTAPIInst.runRT();
  return true;
}

bool initConfigMapByDict(std::map<std::string, std::string>& config_dict, std::map<std::string, std::any>& config_map);
bool initConfigMapByJSON(const std::string& config, std::map<std::string, std::any>& config_map);

bool initRT(std::string& config, std::map<std::string, std::string>& config_dict)
{
  std::map<std::string, std::any> config_map;

  bool pass = false;
  pass = !pass ? initConfigMapByJSON(config, config_map) : pass;
  pass = !pass ? initConfigMapByDict(config_dict, config_map) : pass;
  if (!pass) {
    return false;
  }
  RTAPIInst.initRT(config_map);
  return true;
}

}  // namespace python_interface
