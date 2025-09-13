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

#include <string>

#include "init_sta.hh"

namespace python_interface {

bool layout_patchs(const std::string& path);
bool layout_graph(const std::string& path);
bool generate_vectors(std::string dir, int patch_row_step, int patch_col_step, bool batch_mode);
bool read_vectors_nets(std::string dir);
bool read_vectors_nets_patterns(std::string path);

// for vectorization wire timing graph.
ieval::TimingWireGraph get_timing_wire_graph(std::string wire_graph_path);
ieval::TimingInstanceGraph get_timing_instance_graph(std::string instance_graph_path);

}  // namespace python_interface