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
#include <unordered_map>

#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsSignalWire.hh"

namespace icts {

class OptiNet {
 public:
  typedef int id_type;

  OptiNet(CtsNet *clk_net);
  ~OptiNet() = default;

  CtsNet *get_clk_net() const { return _clk_net; }
  CtsInstance *get_driver() const { return _clk_net->get_driver_inst(); }
  vector<CtsInstance *> get_loads() const { return _clk_net->get_load_insts(); }
  vector<CtsSignalWire> get_signal_wires() const {
    return _clk_net->get_signal_wires();
  }

  int getId(const std::string &name) const {
    if (_name_to_id.count(name) != 0) {
      return _name_to_id.at(name);
    }
    return -1;
  }

 private:
  CtsNet *_clk_net;
  std::unordered_map<std::string, id_type> _name_to_id;
};

}  // namespace icts