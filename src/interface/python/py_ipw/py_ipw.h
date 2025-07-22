// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <map>
#include <set>
#include <string>

#include "api/PowerEngine.hh"

namespace python_interface {
bool readRustVCD(const char* vcd_path, const char* top_instance_name);
bool read_pg_spef(std::string pg_spef_file);

unsigned reportPower();
unsigned report_ir_drop(std::vector<std::string> power_nets);

// for dataflow.
unsigned create_data_flow();

std::map<std::size_t, std::vector<ipower::ClusterConnection>>
build_connection_map(std::vector<std::set<std::string>> clusters,
                     std::set<std::string> src_instances, unsigned max_hop);

std::vector<ipower::MacroConnection> build_macro_connection_map(unsigned max_hop);

}  // namespace python_interface
