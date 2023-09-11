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

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "CtsConfig.h"
#include "CtsDesign.hh"
#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsSignalWire.hh"
#include "EvalNet.h"
#include "pgl.h"
namespace icts {
class Router
{
 public:
  Router() = default;
  Router(const Router&) = default;
  ~Router() = default;
  void init();
  void update();
  void build();

 private:
  void printLog();
  void gocaRouting(CtsNet* clk_net);
  void breakLongWire(CtsNet* clk_net);
  bool routeAble(CtsNet* clk_net);
  std::vector<CtsPin*> getSinkPins(CtsNet* clk_net);
  std::vector<CtsPin*> getBufferPins(CtsNet* clk_net);
  void removeSinkPin(CtsNet* clk_net);

  std::vector<CtsClock*> _clocks;
  std::vector<CtsNet*> _insert_nets;
  std::unordered_map<std::string, CtsInstance*> _name_to_inst;
};

}  // namespace icts