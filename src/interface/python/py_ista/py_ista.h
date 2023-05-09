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
namespace python_interface {
bool staRun(const std::string& output);

bool staInit(const std::string& output);

bool staReport(const std::string& output);
bool setDesignWorkSpace(const std::string& design_workspace);

bool readVerilog(const std::string& file_name);

bool readLiberty(const std::string& file_name);

bool linkDesign(const std::string& cell_name);

bool readSpef(const std::string& file_name);

bool readSdc(const std::string& file_name);
bool reportTiming(int digits, const std::string& delay_type, std::set<std::string> exclude_cell_names, bool derate);

}  // namespace python_interface