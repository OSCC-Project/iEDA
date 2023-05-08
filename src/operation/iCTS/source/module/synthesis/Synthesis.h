#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "ClockTopo.h"
#include "CtsConfig.h"
#include "CtsDBWrapper.h"
#include "CtsDesign.h"
#include "CtsInstance.h"
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

 private:
  void printLog();

 private:
  icts::Placer *_placer = nullptr;
  vector<CtsNet *> _nets;
};
}  // namespace icts