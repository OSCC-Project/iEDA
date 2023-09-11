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
#include "bound_skew_tree/GeomCalc.hh"
#include "gtest/gtest.h"
#include "log/Log.hh"
// debug
#include "model/mplHelper/MplHelper.h"

using ieda::Log;

namespace {
using icts::BST;
using icts::Inst;
using icts::LayerPattern;
using icts::Net;
using icts::Node;
using icts::Pin;
using icts::Point;
using icts::RCPattern;
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

void resetNet(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto load_pins = net->get_load_pins();
  std::vector<Node*> to_be_removed;
  auto find_steiner = [&to_be_removed](Node* node) {
    if (node->isSteiner()) {
      to_be_removed.push_back(node);
    }
  };
  driver_pin->preOrder(find_steiner);
  // recover load pins' timing
  std::ranges::for_each(load_pins, [](Pin* pin) {
    pin->set_parent(nullptr);
    pin->set_children({});
    pin->set_slew_in(0);
    pin->set_cap_load(0);
    pin->set_net(nullptr);
  });
  // release buffer and its pins
  auto* buffer = driver_pin->get_inst();
  delete buffer;
  // release steiner node
  std::ranges::for_each(to_be_removed, [](Node* node) { delete node; });
}

void bstTest(const std::vector<Pin*>& load_pins, const Point& guide_loc)
{
  auto roots = TreeBuilder::dmeTree("bst", load_pins, 0.08, guide_loc);
  LOG_FATAL_IF(roots.size() != 1) << "Case not processed yet!";
  auto root = roots.front();
  auto* driver_pin = root->get_driver_pin();
  auto* net = driver_pin->get_net();

  TimingPropagator::update(net);
  TreeBuilder::writePy(driver_pin, "BST_old");
  LOG_INFO << "BST_old";
  LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
  LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
  LOG_INFO << "max delay: " << driver_pin->get_max_delay();
  LOG_INFO << "max delay(Not Cell): " << driver_pin->get_max_delay() - root->get_insert_delay();
  LOG_INFO << "guide gap: " << TimingPropagator::calcLen(guide_loc, driver_pin->get_location());
  resetNet(net);
}

void boundSkewTreeTest(const std::vector<Pin*>& load_pins, const Point& guide_loc)
{
  auto* root = TreeBuilder::boundSkewTree("BoundSkewTree", load_pins, 0.08, guide_loc);

  auto* driver_pin = root->get_driver_pin();
  auto* net = driver_pin->get_net();

  TimingPropagator::update(net);
  TreeBuilder::writePy(driver_pin, "BoundSkewTree");
  LOG_INFO << "BoundSkewTree";
  LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
  LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
  LOG_INFO << "max delay: " << driver_pin->get_max_delay();
  LOG_INFO << "max delay(Not Cell): " << driver_pin->get_max_delay() - root->get_insert_delay();
  LOG_INFO << "guide gap: " << TimingPropagator::calcLen(guide_loc, driver_pin->get_location());
  resetNet(net);
}

void saltTest(const std::vector<Pin*>& load_pins, const Point& guide_loc)
{
  auto root = TreeBuilder::genBufInst("root", guide_loc);
  root->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  auto* driver_pin = root->get_driver_pin();
  TreeBuilder::shallowLightTree(driver_pin, load_pins);
  auto* net = TimingPropagator::genNet("salt", driver_pin, load_pins);
  TimingPropagator::update(net);

  TreeBuilder::writePy(driver_pin, "SALT");
  LOG_INFO << "SALT";
  LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
  LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
  LOG_INFO << "max delay: " << driver_pin->get_max_delay();
  LOG_INFO << "max delay(Not Cell): " << driver_pin->get_max_delay() - root->get_insert_delay();
  LOG_INFO << "guide gap: " << TimingPropagator::calcLen(guide_loc, driver_pin->get_location());
  resetNet(net);
}

TEST_F(SolverTest, Compare)
{
  dmInst->init("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/db_default_config.json");
  CTSAPIInst.init("/home/liweiguo/project/iEDA/scripts/salsa20/iEDA_config/cts_default_config.json");
  LOG_INFO << "\n\n\n";
  LOG_INFO << "Router unit res (H): " << CTSAPIInst.getClockUnitRes(LayerPattern::kH);
  LOG_INFO << "Router unit cap (H): " << CTSAPIInst.getClockUnitCap(LayerPattern::kH);
  LOG_INFO << "Router unit res (V): " << CTSAPIInst.getClockUnitRes(LayerPattern::kV);
  LOG_INFO << "Router unit cap (V): " << CTSAPIInst.getClockUnitCap(LayerPattern::kV);
  // build tree
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
    auto* load_pin = buf->get_load_pin();
    auto pattern = static_cast<RCPattern>(1 + std::rand() % 2);
    load_pin->set_pattern(pattern);
  }
  std::vector<Pin*> load_pins;
  std::transform(load_bufs.begin(), load_bufs.end(), std::back_inserter(load_pins), [](Inst* buf) { return buf->get_load_pin(); });
  auto guide_loc = Point(99000, 154000);

  bstTest(load_pins, guide_loc);
  boundSkewTreeTest(load_pins, guide_loc);
  saltTest(load_pins, guide_loc);
}

TEST_F(SolverTest, GeomTest)
{
  using icts::bst::GeomCalc;
  using icts::bst::Pt;
  std::vector<Pt> poly = {Pt(0, 0), Pt(0, 0.5), Pt(0, 1), Pt(1, 1), Pt(1, 0)};
  GeomCalc::convexHull(poly);
  EXPECT_EQ(poly.size(), 4);
}

}  // namespace