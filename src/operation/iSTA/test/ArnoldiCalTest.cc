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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "Type.hh"
#include "delay/ArnoldiDelayCal.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "gtest/gtest.h"
#include "liberty/Liberty.hh"
#include "log/Log.hh"
#include "netlist/Net.hh"
#include "netlist/Netlist.hh"
#include "netlist/Pin.hh"
#include "spef/parser-spef.hpp"
#include "sta/Sta.hh"
using ieda::Log;
using ieda::BTreeMap;
using ista::ArnoldiNet;
using ista::DesignObject;
using ista::Liberty;
using ista::Net;
using ista::NetIterator;
using ista::Netlist;
using ista::NetPinIterator;
using ista::RcNet;
using ista::Sta;

namespace {

class ArnoldiCal : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(ArnoldiCal, delay_cal) {
  Sta* ista = Sta::getOrCreateSta();

  Liberty lib;
  auto load_lib =

      lib.loadLiberty(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core_Early.lib");

  LOG_INFO << "build lib test";

  ista->addLib(std::move(load_lib));

  ista->readVerilog("/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.v");
  ista->linkDesign("aes_core");

  Netlist* design_nl = ista->get_netlist();

  spef::Spef parser;  // create a parser object

  if (parser.read(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.spef")) {
  } else {
    std::cout << "error";  // show the error message
  }

  try {
    Net* net;
    FOREACH_NET(design_nl, net) {
      const char* net_name = net->get_name();
      std::string rc_net_name = net_name;
      if (rc_net_name == "net_8058") {
        auto arnoldi_net = std::make_unique<ArnoldiNet>(net);
        DesignObject* driver = net->getDriver();
        std::string load_name;
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

        std::vector<double> current = {0.0547, 0.1066, 0.1650, 0.2150, 0.2490,
                                       0.2554, 0.2201, 0.1604, 0.1012, 0.0664};

        // double time = 0.02575 * 1.0e-6;
        double arnoldi_delay = 0.0;
        auto* spef_net = parser.findSpefNet(rc_net_name);
        LOG_FATAL_IF(!spef_net);
        arnoldi_net->updateRcTiming(*spef_net);
        //    arnoldi_delay = arnoldi_net->get_delay(
        //    current, sim_total_time, num_sim_point, ista::TransType::kRise);

        std::cout << arnoldi_delay << std::endl;

        std::cout << driver->getFullName() << " ----> " << load_name
                  << " delay is " << arnoldi_delay << std::endl;
        std::cout << "--------------------------------" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  std::cout << "test finish" << std::endl;
}

TEST_F(ArnoldiCal, delay_cal_de) {
  std::vector<double> current = {0.0547, 0.1066, 0.1650, 0.2150, 0.2490,
                                 0.2554, 0.2201, 0.1604, 0.1012, 0.0664};
  //double sim_total_time = 0.2575 * 1.0e-6;
  // int num_sim_point = 10;
  //  double time = 0.02575 * 1.0e-6;
  // double arnoldi_delay;
  // double arnoldi_slew;
  // double vdd = 1.1;

  ista::ArnoldiNet arnoldi_test(nullptr);
  std::vector<double> nodal_capa = {19.5, 21.20};
  arnoldi_test.set_nodal_caps(std::move(nodal_capa));
  std::vector<double> nodal_res_all = {4.140};

  // arnoldi_delay = arnoldi_test.delay(current, sim_total_time, num_sim_point,
  //                                   ista::TransType::kRise);
  // std::cout << arnoldi_delay << std::endl;

  // arnoldi_slew = arnoldi_test.slew(current, sim_total_time, num_sim_point,
  //                                ista::TransType::kRise);
  // std::cout << arnoldi_slew << std::endl;
}
TEST_F(ArnoldiCal, total_test) {
  Sta* ista = Sta::getOrCreateSta();

  Liberty lib;
  auto load_lib =

      lib.loadLiberty(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core_Early.lib");

  LOG_INFO << "build lib test";

  ista->addLib(std::move(load_lib));

  ista->readVerilog("/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.v");
  ista->linkDesign("aes_core");

  Netlist* design_nl = ista->get_netlist();

  spef::Spef parser;  // create a parser object

  if (parser.read(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.spef")) {
  } else {
    std::cout << "error";  // show the error message
  }

  // int cnt = 0;

  try {
    Net* net;
    FOREACH_NET(design_nl, net) {
      const char* net_name = net->get_name();
      std::string rc_net_name = net_name;
      if (rc_net_name == "net_8058") {
        auto arnoldi_net = std::make_unique<ArnoldiNet>(net);
        DesignObject* driver = net->getDriver();
        std::string load_name;
        DesignObject* obj;
        ista::Pin pin("A1", nullptr);
        ista::Instance instance("inst_6212", nullptr);
        pin.set_own_instance(&instance);
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

        double start_time = 0;
        double end_time = 0.2575 * 1.0e-6 * 9;
        // double sim_total_time = 0.2575 * 1.0e-6;
        int num_sim_point = 10;
        // double time = 0.02575 * 1.0e-6;

        // double vdd = 1.1;

        auto get_current = [](double front, double end,
                              int num) -> std::vector<double> {
          std::vector<double> current = {0.0547, 0.1066, 0.1650, 0.2150,
                                         0.2490, 0.2554, 0.2201, 0.1604,
                                         0.1012, 0.0664};
          return current;
        };

        // double coefficient = 0.5;
        auto* spef_net = parser.findSpefNet(rc_net_name);
        LOG_FATAL_IF(!spef_net);
        arnoldi_net->updateRcTiming(*spef_net);
        auto arnoldi_delay = arnoldi_net->getDelay(
            get_current, start_time, end_time, num_sim_point,
            ista::AnalysisMode::kMax, ista::TransType::kRise, &pin);
        auto arnoldi_slew = arnoldi_net->getSlew(
            get_current, start_time, end_time, num_sim_point,
            ista::AnalysisMode::kMax, ista::TransType::kRise, &pin);
        std::cout << driver->getFullName() << " ----> " << load_name
                  << " delay is " << arnoldi_delay->first << std::endl;
        std::cout << driver->getFullName() << " ----> " << load_name
                  << " slew is " << *arnoldi_slew << std::endl;
        std::cout << "--------------------------------" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  std::cout << "test finish" << std::endl;
}

TEST_F(ArnoldiCal, arnoldi) {
  Sta* ista = Sta::getOrCreateSta();

  Liberty lib;
  auto load_lib =

      lib.loadLiberty(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core_Early.lib");

  LOG_INFO << "build lib test";

  ista->addLib(std::move(load_lib));

  ista->readVerilog("/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.v");
  ista->linkDesign("aes_core");

  Netlist* design_nl = ista->get_netlist();

  spef::Spef parser;  // create a parser object

  if (parser.read(
          "/home/liuh/iEDA/src/iSTA/benchmark/aes_core/aes_core.spef")) {
  } else {
    std::cout << "error";  // show the error message
  }

  try {
    Net* net;
    FOREACH_NET(design_nl, net) {
      const char* net_name = net->get_name();
      std::string rc_net_name = net_name;
      if (rc_net_name == "net_8058") {
        auto arnoldi_net = std::make_unique<ArnoldiNet>(net);
        DesignObject* driver = net->getDriver();
        std::string load_name;
        DesignObject* obj;
        ista::Pin pin("A1", nullptr);
        ista::Instance instance("inst_6212", nullptr);
        pin.set_own_instance(&instance);
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

        double start_time = 0;
        double end_time = 0.2575 * 1.0e-6 * 9;
        // double sim_total_time = 0.2575 * 1.0e-6;
        int num_sim_point = 10;
        // double time = 0.02575 * 1.0e-6;

        //  double vdd = 1.1;

        auto get_current = [](double front, double end,
                              int num) -> std::vector<double> {
          std::vector<double> current = {0.0547, 0.1066, 0.1650, 0.2150,
                                         0.2490, 0.2554, 0.2201, 0.1604,
                                         0.1012, 0.0664};
          return current;
        };

        // double coefficient = 0.5;
        auto* spef_net = parser.findSpefNet(rc_net_name);
        LOG_FATAL_IF(!spef_net);
        arnoldi_net->updateRcTiming(*spef_net);
        auto arnoldi_delay = arnoldi_net->getDelay(
            get_current, start_time, end_time, num_sim_point,
            ista::AnalysisMode::kMax, ista::TransType::kRise, &pin);
        auto arnoldi_slew = arnoldi_net->getSlew(
            get_current, start_time, end_time, num_sim_point,
            ista::AnalysisMode::kMax, ista::TransType::kRise, &pin);
        std::cout << driver->getFullName() << " ----> " << load_name
                  << " delay is " << arnoldi_delay->first << std::endl;
        std::cout << driver->getFullName() << " ----> " << load_name
                  << " slew is " << *arnoldi_slew << std::endl;
        std::cout << "--------------------------------" << std::endl;

        // int num_nodes = arnoldi_net->_nodal_caps.size();
        // Eigen::VectorXd nodal_capac(num_nodes);
        // for (int i = 0; i < num_nodes; ++i) {
        //   nodal_capac(i) = arnoldi_net->_nodal_caps[i] * 1.0e2;
        // }
        // Eigen::MatrixXd cap_dia_mat = nodal_capac.asDiagonal();
        // std::cout << cap_dia_mat << std::endl;

        // auto C = (arnoldi_net->_conductances_matrix) * 100000;
        // std::cout << C << std::endl;

        // auto B = C.inverse();
        // std::cout << B << std::endl;

        // Eigen::MatrixXd A = B * cap_dia_mat;

        // int r = A.cols();
        // int c = A.rows();

        // std::cout << A << std::endl;
        // std::cout << r << " " << c << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  std::cout << "test finish" << std::endl;
}
}  // namespace
