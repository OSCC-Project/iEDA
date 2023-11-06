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
#include "liberty/Liberty.hh"
#include "log/Log.hh"
#include "netlist/Net.hh"
#include "netlist/Netlist.hh"
#include "spef/parser-spef.hpp"
#include "sta/Sta.hh"

using ieda::Log;
using ista::DesignObject;
using ista::Liberty;
using ista::Net;
using ista::NetIterator;
using ista::Netlist;
using ista::NetPinIterator;
using ista::RcNet;
using ista::Sta;

namespace {

class DelayTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(DelayTest, test1) {
  spef::Spef parser;  // create a parser object
  if (parser.read("/home/liuh/iEDA/src/iSTA/examples/test.spef")) {  // parse a
    // .spef
    std::cout << parser.dump() << '\n';  // dump the parsed spef
  } else {
    std::cout << "error";  // show the error message
  }
  std::cout << "******************PRINT****************" << std::endl;
  for (auto net : parser.nets) {
    std::cout << "print name、type、direction" << std::endl;
    for (auto conn : net.connections) {
      std::cout << conn.type << " " << conn.name << " " << conn.direction << " "
                << std::endl;
    }

    std::cout << "print coordinate" << std::endl;
    for (auto conn : net.connections) {
      auto co1 = std::get<0>(*(conn.coordinate));
      auto co2 = std::get<1>(*(conn.coordinate));
      std::cout << co1 << " ";
      std::cout << co2 << " " << std::endl;
    }
    std::cout << "print connections " << std::endl;
    for (const auto& c : net.connections) {
      std::cout << c << '\n';
    }

    std::cout << "print load" << std::endl;
    for (auto conn : net.connections) {
      if (bool(conn.load)) {
        std::cout << *(conn.load) << std::endl;
      }
    }
    std::cout << "print capacitances" << std::endl;
    for (auto cap : net.caps) {
      std::string c1 = std::get<0>(cap);
      std::cout << c1 << " ";

      std::string c2 = std::get<1>(cap);
      std::cout << c2 << " ";

      float c3 = std::get<2>(cap);
      std::cout << c3 << std::endl;
    }
    std::cout << "print resistances" << std::endl;
    for (auto res : net.ress) {
      std::string r1 = std::get<0>(res);
      std::cout << r1 << " ";

      std::string r2 = std::get<1>(res);
      std::cout << r2 << " ";

      float r3 = std::get<2>(res);
      std::cout << r3 << std::endl;
    }
  }
}
TEST_F(DelayTest, updateTiming) {
  Sta* ista = Sta::getOrCreateSta();

  Liberty lib;
  auto load_lib =
      lib.loadLiberty("/home/liuh/iEDA/src/iSTA/examples/example1_fast.lib");
  LOG_INFO << "build lib test";

  EXPECT_TRUE(load_lib);
  ista->addLib(std::move(load_lib));

  ista->readVerilog("/home/liuh/iEDA/src/iSTA/examples/example1.v");

  ista->linkDesign("top");

  Netlist* design_nl = ista->get_netlist();

  spef::Spef parser;  // create a parser object
  if (parser.read("/home/liuh/iEDA/src/iSTA/examples/test.spef")) {
    std::cout << parser.dump() << '\n';  // dump the parsed spef
  } else {
    std::cout << "error";  // show the error message
  }

  Net* net;
  FOREACH_NET(design_nl, net) {
    const char* net_name = net->get_name();
    std::string rc_net_name = net_name;
    RcNet* rc_net = new RcNet(net);
    DesignObject* driver = net->getDriver();

    auto* spef_net = parser.findSpefNet(rc_net_name);
    LOG_FATAL_IF(!spef_net);

    // rc_net->updateRcTiming(*spef_net);
    std::string load_name = "";
    DesignObject* obj;
    FOREACH_NET_PIN(net, obj) {
      if (obj->getFullName() != driver->getFullName()) {
        load_name = obj->getFullName();
      }
    }
    std::cout << "Net name:" << net->get_name() << std::endl;
    std::cout << "Driver:"
              << " " << driver->getFullName() << std::endl;
    std::cout << "Load:"
              << " " << load_name << std::endl;
    double load_delay = 0.0f;
    ista::RcTree* rct = rc_net->rct();
    auto node = rct->node(load_name);
    load_delay = node->delay();
    std::cout << driver->getFullName() << " ----> " << load_name << " delay is "
              << load_delay << std::endl;
    std::cout << "--------------------------------" << std::endl;
  }
}

TEST_F(DelayTest, updateTiming1) {
  Sta* ista = Sta::getOrCreateSta();

  Liberty lib;
  auto load_lib =

      lib.loadLiberty(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core_Early.lib");

  LOG_INFO << "build lib test";

  ista->addLib(std::move(load_lib));

  ista->readVerilog("/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.v");
  ista->linkDesign("top");

  Netlist* design_nl = ista->get_netlist();

  spef::Spef parser;  // create a parser object

  if (parser.read(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.spef")) {
  } else {
    std::cout << "error";  // show the error message
  }

  int cnt = 0;

  try {
    Net* net;
    FOREACH_NET(design_nl, net) {
      const char* net_name = net->get_name();
      std::string rc_net_name = net_name;
      // if (rc_net_name == "net_8058") {
      RcNet* rc_net = new RcNet(net);
      DesignObject* driver = net->getDriver();
      std::string load_name = "";
      DesignObject* obj;
      FOREACH_NET_PIN(net, obj) {
        if (obj->getFullName() != driver->getFullName()) {
          load_name = obj->getFullName();
        }
      }
      std::cout << "--------------------------------" << std::endl;
      std::cout << "Net name:" << net->get_name() << std::endl;
      std::cout << "Driver:"
                << " " << driver->getFullName() << std::endl;
      std::cout << "Load:"
                << " " << load_name << std::endl;
      std::cout << "--------------------------------" << std::endl;

      auto* spef_net = parser.findSpefNet(rc_net_name);
      LOG_FATAL_IF(!spef_net);

      // rc_net->updateRcTiming(*spef_net);

      double load_delay = 0.0f;
      ista::RcTree* rct = rc_net->rct();
      auto node = rct->node(load_name);
      load_delay = node->delay();
      std::cout << driver->getFullName() << " ----> " << load_name
                << " delay is " << load_delay << std::endl;
      std::cout << "--------------------------------" << std::endl;
      delete rc_net;
      //}
      cnt++;

      if (cnt > 5) {
        break;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  std::cout << "test finish" << std::endl;
  std::cout << "test" << std::endl;
}

TEST_F(DelayTest, nutshell) {
  spef::Spef parser;
  if (!parser.read("/home/smtao/nutshell/soc_asic_top.cmax.125c.spef")) {
    LOG_FATAL << "Parse the spef file error.";
  }
}
}  // namespace