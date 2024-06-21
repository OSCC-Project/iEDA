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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "Type.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "delay/ReduceDelayCal.hh"
#include "gtest/gtest.h"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Net.hh"
#include "netlist/Netlist.hh"
#include "netlist/Pin.hh"
#include "sta/Sta.hh"
using ieda::BTreeMap;
using ieda::Log;
using ista::ArnoldiNet;
using ista::DesignObject;
using ista::Lib;
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

}  // namespace
