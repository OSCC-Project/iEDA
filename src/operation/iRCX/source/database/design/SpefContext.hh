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

#include <map>
#include <string>
#include <vector>

#include "Types.hh"
namespace ircx {

// SpefContext
//
// SPEF output metadata: compressed name tables and instance-to-cell mapping.
// Previously stored inside Database; now a standalone value type populated
// by the layout adapter.
//
struct SpefContext {
  void clear() {
    net_names.clear();
    port_names.clear();
    port_io.clear();
    instance_names.clear();
    instance_to_cell.clear();
  }

  // Ordered lists for SPEF *NAME_MAP output.
  // Index 0 = first entry; SPEF IDs start at *1.
  std::vector<Str> net_names;
  std::vector<Str> port_names;
  std::vector<char> port_io;
  std::vector<Str> instance_names;

  // instance name → cell name (for *CELL entries)
  std::map<Str, Str> instance_to_cell;
};

}  // namespace ircx
