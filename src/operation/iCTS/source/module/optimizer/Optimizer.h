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

#include "CtsDBWrapper.h"
#include "OptiNet.h"
#include "Router.h"

namespace icts {

class Optimizer {
 public:
  typedef std::vector<CtsNet *>::iterator NetIterator;
  typedef std::vector<IdbNet *>::iterator IdbNetIterator;

  Optimizer() = default;
  Optimizer(const Optimizer &optimizer) = delete;
  ~Optimizer() = default;

  void optimize(NetIterator begin, NetIterator end);
  void update();

 private:

  CtsInstance *get_cts_inst(IdbInstance *idb_inst) const;
  IdbNet *get_idb_net(const OptiNet &opti_net) const;
  Point get_location(IdbInstance *idb_inst) const;

 private:
  std::vector<IdbNet *> _idb_nets;
  std::vector<std::vector<CtsSignalWire>> _topos;
};

}  // namespace icts