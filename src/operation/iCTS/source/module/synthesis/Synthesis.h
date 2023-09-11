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

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "ClockTopo.h"
#include "CtsConfig.h"
#include "CtsDBWrapper.h"
#include "CtsDesign.hh"
#include "CtsInstance.hh"
#include "Placer.h"

namespace icts {
using std::string;

class Synthesis {
 public:
  Synthesis() { _placer = new Placer(); }
  ~Synthesis() = default;

  void init();
  void insertCtsNetlist();
  void incrementalInsertCtsNetlist();
  void insertInstance(CtsInstance *inst);
  void insertInstance(ClockTopo &clk_topo);
  void insertNet(ClockTopo &clk_topo);
  void incrementalInsertInstance(ClockTopo &clk_topo);
  void incrementalInsertNet(ClockTopo &clk_topo);
  void update();
  void place(CtsInstance *inst);
  void cancelPlace(CtsInstance *inst);

 private:
  void printLog();

 private:
  icts::Placer *_placer = nullptr;
  vector<CtsNet *> _nets;
};
}  // namespace icts