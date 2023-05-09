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
#include "OptiNet.h"

#include <set>

namespace icts {

OptiNet::OptiNet(CtsNet *clk_net) : _clk_net(clk_net) {
  std::set<std::string> node_names;
  auto &signal_wires = clk_net->get_signal_wires();
  for (const auto &signal_wire : signal_wires) {
    node_names.insert(signal_wire.get_first()._name);
    node_names.insert(signal_wire.get_second()._name);
  }
  id_type id = 0;
  for (auto itr = node_names.begin(); itr != node_names.end(); ++itr) {
    if (_name_to_id.count(*itr) == 0) {
      _name_to_id.insert(std::make_pair(*itr, id++));
    }
  }
}

}  // namespace icts