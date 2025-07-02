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
#include <iostream>
#include <string>
#include <utility>

#include "delay/ElmoreDelayCalc.hh"
#include "gtest/gtest.h"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Net.hh"
#include "netlist/Netlist.hh"
#include "sta/Sta.hh"
#include "api/TimingEngine.hh"

using ieda::Log;
using ista::DesignObject;
using ista::Lib;
using ista::Net;
using ista::NetIterator;
using ista::Netlist;
using ista::NetPinIterator;
using ista::RcNet;
using ista::Sta;

using namespace ista;
using ieda::Log;
using ieda::Stats;

namespace {

class DelayTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(DelayTest, virtual_rc_tree) {
  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();

  auto& virutal_rc_tree = timing_engine->initVirtualRcTree("virtual_rc_tree");

  auto* root_node = timing_engine->makeOrFindVirtualRCTreeNode("virtual_rc_tree", "root");
  root_node->setCap(0.1);
  virutal_rc_tree.set_root(root_node);

  auto* inner_node = timing_engine->makeOrFindVirtualRCTreeNode("virtual_rc_tree", "inner");
  inner_node->setCap(0.2);

  timing_engine->makeVirtualRCTreeResistor("virtual_rc_tree", root_node, inner_node, 10);

  auto* leaf_node = timing_engine->makeOrFindVirtualRCTreeNode("virtual_rc_tree", "leaf");
  leaf_node->setCap(0.3);

  timing_engine->makeVirtualRCTreeResistor("virtual_rc_tree", inner_node, leaf_node, 10);

  timing_engine->updateVirtualRCTreeInfo("virtual_rc_tree");

  auto node_delays = timing_engine->getVirtualRCTreeAllNodeDelay("virtual_rc_tree");
  for (auto& [node_name, delay] : node_delays) {
    std::cout << node_name << ": " << delay << std::endl;
  }

  auto node_slews = timing_engine->getVirtualRCTreeAllNodeSlew("virtual_rc_tree", 0.002);
  for (auto& [node_name, slew] : node_slews) {
    std::cout << node_name << ": " << slew << std::endl;
  }
}

}  // namespace