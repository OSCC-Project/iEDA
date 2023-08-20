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
#include <vector>

#include "../../database/interaction/ids.hpp"
#include "../../platform/data_manager/idm.h"
#include "CTSAPI.hpp"
#include "Inst.hh"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "bound_skew_tree/BST.hh"
#include "bound_skew_tree/BoundSkewTree.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"

using ieda::Log;

namespace {
using icts::BST;
using icts::Inst;
using icts::Pin;
using icts::Point;
using icts::TimingPropagator;
using icts::TreeBuilder;
class SolverTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

void boundSkewTreeTest()
{
  auto loc_list
      = std::vector<Point>{// Point(122000, 196000),
                           Point(128000, 154000), Point(90000, 54000),  Point(84000, 158000), Point(98000, 186000), Point(74000, 98000),
                           Point(108000, 146000), Point(134000, 60000), Point(80000, 198000), Point(176000, 54000), Point(128000, 150000),
                           Point(108000, 150000), Point(98000, 158000), Point(98000, 196000), Point(134000, 54000)};
  std::vector<Inst*> load_bufs;
  for (size_t i = 0; i < loc_list.size(); ++i) {
    auto loc = loc_list[i];
    auto* buf = TreeBuilder::genBufInst(CTSAPIInst.toString("buf_", i), loc);
    buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
    load_bufs.push_back(buf);
  }
  std::vector<Pin*> pins;
  std::ranges::for_each(load_bufs, [&pins](Inst* buf) { pins.emplace_back(buf->get_load_pin()); });
  auto guide_loc = Point(122000, 196000);
  auto roots = TreeBuilder::boundSkewTree("BoundSkewTree", pins, 0.08, guide_loc);
  if (roots.size() == 1) {
    auto root = roots.front();
    auto* driver_pin = root->get_driver_pin();
    auto* net = driver_pin->get_net();
    std::ranges::for_each(pins, [](Pin* pin) {
      pin->set_max_delay(0);
      pin->set_min_delay(0);
    });
    TimingPropagator::netLenPropagate(net);
    TimingPropagator::capPropagate(net);
    TimingPropagator::wireDelayPropagate(net);
    TreeBuilder::writePy(driver_pin, "BoundSkewTree");
    LOG_INFO << "BoundSkewTree";
    LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
    LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
    LOG_INFO << "max delay: " << driver_pin->get_max_delay();
  }
}

void bstTest()
{
  auto loc_list
      = std::vector<Point>{// Point(122000, 196000),
                           Point(128000, 154000), Point(90000, 54000),  Point(84000, 158000), Point(98000, 186000), Point(74000, 98000),
                           Point(108000, 146000), Point(134000, 60000), Point(80000, 198000), Point(176000, 54000), Point(128000, 150000),
                           Point(108000, 150000), Point(98000, 158000), Point(98000, 196000), Point(134000, 54000)};
  std::vector<Inst*> load_bufs;
  for (size_t i = 0; i < loc_list.size(); ++i) {
    auto loc = loc_list[i];
    auto* buf = TreeBuilder::genBufInst(CTSAPIInst.toString("buf_", i), loc);
    buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
    load_bufs.push_back(buf);
  }
  std::vector<Pin*> pins;
  std::ranges::for_each(load_bufs, [&pins](Inst* buf) { pins.emplace_back(buf->get_load_pin()); });
  auto guide_loc = Point(122000, 196000);
  auto roots = TreeBuilder::dmeTree("bst", pins, 0.08, guide_loc);
  if (roots.size() == 1) {
    auto root = roots.front();
    auto* driver_pin = root->get_driver_pin();
    auto* net = driver_pin->get_net();
    std::ranges::for_each(pins, [](Pin* pin) {
      pin->set_max_delay(0);
      pin->set_min_delay(0);
    });
    TimingPropagator::netLenPropagate(net);
    TimingPropagator::capPropagate(net);
    TimingPropagator::wireDelayPropagate(net);
    TreeBuilder::writePy(driver_pin, "BST");
    LOG_INFO << "BST";
    LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
    LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
    LOG_INFO << "max delay: " << driver_pin->get_max_delay();
  }
}

void saltTest()
{
  auto loc_list
      = std::vector<Point>{// Point(122000, 196000),
                           Point(128000, 154000), Point(90000, 54000),  Point(84000, 158000), Point(98000, 186000), Point(74000, 98000),
                           Point(108000, 146000), Point(134000, 60000), Point(80000, 198000), Point(176000, 54000), Point(128000, 150000),
                           Point(108000, 150000), Point(98000, 158000), Point(98000, 196000), Point(134000, 54000)};
  std::vector<Inst*> load_bufs;
  for (size_t i = 0; i < loc_list.size(); ++i) {
    auto loc = loc_list[i];
    auto* buf = TreeBuilder::genBufInst(CTSAPIInst.toString("buf_", i), loc);
    buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
    load_bufs.push_back(buf);
  }
  std::vector<Pin*> pins;
  std::ranges::for_each(load_bufs, [&pins](Inst* buf) { pins.emplace_back(buf->get_load_pin()); });
  auto driver_buf = TreeBuilder::genBufInst("root", Point(106960, 132400));
  driver_buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  auto* driver_pin = driver_buf->get_driver_pin();

  TreeBuilder::shallowLightTree(driver_pin, pins);
  auto* net = TimingPropagator::genNet("salt", driver_pin, pins);
  TimingPropagator::netLenPropagate(net);
  TimingPropagator::capPropagate(net);
  TimingPropagator::wireDelayPropagate(net);

  TreeBuilder::writePy(driver_pin, "SALT");
  LOG_INFO << "SALT";
  LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
  LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
  LOG_INFO << "max delay: " << driver_pin->get_max_delay() - driver_buf->get_insert_delay();
}

TEST_F(SolverTest, Compare)
{
  dmInst->init("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json");
  CTSAPIInst.init("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  // bstTest();
  // saltTest();
}

}  // namespace