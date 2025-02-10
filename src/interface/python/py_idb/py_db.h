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

#include <set>
#include <string>
#include <vector>

namespace python_interface {

bool initIdb(const std::string& config_path);
bool initTechLef(const std::string& techlef_path);
bool initLef(const std::vector<std::string>& lef_paths);
bool initDef(const std::string& def_path);
bool initVerilog(const std::string& verilog_path, const std::string& top_module);
bool saveDef(const std::string& def_name);
bool saveNetList(const std::string& netlist_path, std::set<std::string> exclude_cell_names = {}, bool is_add_space_for_escape_name = false);
bool saveGDSII(const std::string& gds_name);

}  // namespace python_interface