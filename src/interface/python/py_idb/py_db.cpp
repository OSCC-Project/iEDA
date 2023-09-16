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
#include "py_db.h"

#include <idm.h>

namespace python_interface {

bool initIdb(const std::string& config_path)
{
  return dmInst->init(config_path);
}

bool initTechLef(const std::string& techlef_path)
{
  dmInst->get_config().set_tech_lef_path(techlef_path);
  return dmInst->readLef(vector<string>{techlef_path}, true);
}

bool initLef(const std::vector<std::string>& lef_paths)
{
  dmInst->get_config().set_lef_paths(lef_paths);
  return dmInst->readLef(lef_paths);
}

bool initDef(const std::string& def_path)
{
  dmInst->get_config().set_def_path(def_path);
  return dmInst->readDef(def_path);
}

bool initVerilog(const std::string& verilog_path)
{
  dmInst->get_config().set_verilog_path(verilog_path);
  return dmInst->readVerilog(verilog_path);
}

bool saveDef(const std::string& def_name)
{
  return dmInst->saveDef(def_name);
}

bool saveNetList(const std::string& netlist_path, std::set<std::string> exclude_cell_names /* = {} */)
{
  dmInst->saveVerilog(netlist_path, std::move(exclude_cell_names));
  return true;
}

bool saveGDSII(const std::string& gds_name)
{
  return dmInst->saveGDSII(gds_name);
}

}  // namespace python_interface