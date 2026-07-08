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
/**
 * @file FlowClockTreeMaterializationTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief iDB clock-tree materialization and restore tests for Flow.
 */

#include <string>
#include <utility>
#include <vector>

#include "Clock.hh"
#include "Design.hh"
#include "Flow.hh"
#include "FlowDesignFixture.hh"
#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbTerm.h"
#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "common/CTSTestRuntime.hh"
#include "io/Wrapper.hh"

namespace icts_test {
namespace {

using namespace flow_test;

TEST(FlowTest, ClockTreeMaterializationFailureRemovesCreatedIdbNetsAfterFailedCommit)
{
  ScopedFlowReset scoped_flow_reset;
  idb::IdbDesign idb_design;
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_design(&idb_design);

  auto* clock = icts_test::runtime::CurrentRuntime().design.makeClock("LOGICAL_CLK", "cts_inserted_clk_net");
  ASSERT_NE(clock, nullptr);
  BuildClockForWrapperClockTreeMaterialization(*clock, "LOGICAL_CLK_SRC", "restore_ff", "CLK");
  ASSERT_EQ(idb_design.get_net_list()->find_net("cts_inserted_clk_net"), nullptr);

  const auto result = icts_test::runtime::CurrentRuntime().wrapper.writeClocksDetailed(
      icts_test::runtime::CurrentRuntime().design, icts_test::runtime::CurrentRuntime().reporter, {clock});

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.failed_clock, "LOGICAL_CLK");
  EXPECT_EQ(result.failed_net, "cts_inserted_clk_net");
  EXPECT_EQ(result.reason, "write_clock_failed");
  EXPECT_TRUE(result.idb_clock_tree_restored);
  EXPECT_EQ(idb_design.get_net_list()->find_net("cts_inserted_clk_net"), nullptr);
}

TEST(FlowTest, WrapperReadClocksBuildsCtsClockFromSdcDeclaredIdbNet)
{
  ScopedFlowReset scoped_flow_reset;
  idb::IdbLayout idb_layout;
  idb::IdbDesign idb_design(&idb_layout);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_design(&idb_design);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_layout(&idb_layout);

  auto* src_master = AddIdbCellMaster(idb_layout, "SRC_CELL");
  ASSERT_NE(src_master, nullptr);
  ASSERT_NE(AddIdbTerm(*src_master, "CLKOUT", idb::IdbConnectDirection::kOutput, idb::IdbConnectType::kClock), nullptr);
  auto* reg_master = AddIdbCellMaster(idb_layout, "REG_CELL");
  ASSERT_NE(reg_master, nullptr);
  ASSERT_NE(AddIdbTerm(*reg_master, "CLK", idb::IdbConnectDirection::kInput, idb::IdbConnectType::kClock), nullptr);

  auto* src_inst = AddIdbInst(idb_design, "src0", *src_master, 0, 0);
  auto* sink_inst = AddIdbInst(idb_design, "sink0", *reg_master, 100, 0);
  ASSERT_NE(src_inst, nullptr);
  ASSERT_NE(sink_inst, nullptr);
  auto* src_pin = src_inst->get_pin_by_term("CLKOUT");
  auto* sink_pin = sink_inst->get_pin_by_term("CLK");
  ASSERT_NE(src_pin, nullptr);
  ASSERT_NE(sink_pin, nullptr);

  auto* idb_net = idb_design.get_net_list()->add_net("physical_clk_net", idb::IdbConnectType::kClock);
  ASSERT_NE(idb_net, nullptr);
  idb_net->add_instance_pin(src_pin);
  src_pin->set_net(idb_net);
  src_pin->set_net_name("physical_clk_net");
  idb_net->add_instance_pin(sink_pin);
  sink_pin->set_net(idb_net);
  sink_pin->set_net_name("physical_clk_net");

  EXPECT_TRUE(icts_test::runtime::CurrentRuntime().wrapper.readClocks(
      icts_test::runtime::CurrentRuntime().design, icts_test::runtime::CurrentRuntime().reporter, {{"LOGICAL_CLK", "physical_clk_net"}}));
  ASSERT_EQ(icts_test::runtime::CurrentRuntime().design.get_clocks().size(), 1U);
  auto* clock = icts_test::runtime::CurrentRuntime().design.get_clocks().front();
  ASSERT_NE(clock, nullptr);
  EXPECT_EQ(clock->get_clock_name(), "LOGICAL_CLK");
  EXPECT_EQ(clock->get_clock_net_name(), "physical_clk_net");
  ASSERT_NE(clock->get_clock_source(), nullptr);
  ASSERT_NE(clock->get_clock_source_net(), nullptr);
  EXPECT_EQ(clock->get_clock_source()->get_name(), "CLKOUT");
  ASSERT_EQ(clock->get_loads().size(), 1U);
  EXPECT_EQ(clock->get_loads().front()->get_name(), "CLK");
  EXPECT_NE(icts_test::runtime::CurrentRuntime().design.findPin("src0/CLKOUT"), nullptr);
  EXPECT_NE(icts_test::runtime::CurrentRuntime().design.findPin("sink0/CLK"), nullptr);
}

TEST(FlowTest, WrapperClockTreeMaterializationResolvesExistingPinsAndMaterializesCtsBufferInst)
{
  ScopedFlowReset scoped_flow_reset;
  idb::IdbLayout idb_layout;
  idb::IdbDesign idb_design(&idb_layout);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_design(&idb_design);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_layout(&idb_layout);

  auto* reg_master = AddIdbCellMaster(idb_layout, "REG_CELL");
  ASSERT_NE(reg_master, nullptr);
  ASSERT_NE(AddIdbTerm(*reg_master, "CLK", idb::IdbConnectDirection::kInput, idb::IdbConnectType::kClock), nullptr);
  auto* buf_master = AddIdbCellMaster(idb_layout, "CTS_BUF");
  ASSERT_NE(buf_master, nullptr);
  ASSERT_NE(AddIdbTerm(*buf_master, "A", idb::IdbConnectDirection::kInput, idb::IdbConnectType::kClock), nullptr);
  ASSERT_NE(AddIdbTerm(*buf_master, "Y", idb::IdbConnectDirection::kOutput, idb::IdbConnectType::kClock), nullptr);

  auto* sink_inst = AddIdbInst(idb_design, "sink0", *reg_master, 100, 0);
  ASSERT_NE(sink_inst, nullptr);
  auto* idb_sink_pin = sink_inst->get_pin_by_term("CLK");
  ASSERT_NE(idb_sink_pin, nullptr);

  auto* clock = icts_test::runtime::CurrentRuntime().design.makeClock("LOGICAL_CLK", "root_clk_net");
  ASSERT_NE(clock, nullptr);
  auto* source_net = icts_test::runtime::CurrentRuntime().design.makeNet("root_clk_net");
  auto* leaf_net = icts_test::runtime::CurrentRuntime().design.makeNet("leaf_clk_net");
  ASSERT_NE(source_net, nullptr);
  ASSERT_NE(leaf_net, nullptr);
  auto* io_driver = icts_test::runtime::CurrentRuntime().design.makePin("clk_port");
  ASSERT_NE(io_driver, nullptr);
  io_driver->set_name("clk_port");
  io_driver->set_type(icts::PinType::kOut);
  io_driver->set_net(source_net);
  io_driver->set_io(true);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(io_driver));
  auto* buf_inst = MakeDesignInst("cts_buf0", "CTS_BUF", icts::InstType::kBuffer, icts::Point<int>(10, 0));
  ASSERT_NE(buf_inst, nullptr);
  auto* buf_in = AddOwnedLoad(*clock, source_net, *buf_inst, "A");
  ASSERT_NE(buf_in, nullptr);
  auto* buf_out = icts_test::runtime::CurrentRuntime().design.makePin("Y");
  ASSERT_NE(buf_out, nullptr);
  buf_out->set_name("Y");
  buf_out->set_type(icts::PinType::kOut);
  buf_out->set_inst(buf_inst);
  buf_out->set_net(leaf_net);
  buf_inst->insertDriverPin(buf_out);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(buf_out));
  auto* sink_pin = icts_test::runtime::CurrentRuntime().design.makePin("CLK");
  ASSERT_NE(sink_pin, nullptr);
  sink_pin->set_name("CLK");
  sink_pin->set_type(icts::PinType::kClock);
  sink_pin->set_inst(MakeDesignInst("sink0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0)));
  ASSERT_NE(sink_pin->get_inst(), nullptr);
  sink_pin->set_net(leaf_net);
  sink_pin->get_inst()->add_pin(sink_pin);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(sink_pin));

  source_net->set_driver(io_driver);
  source_net->add_load(buf_in);
  leaf_net->set_driver(buf_out);
  leaf_net->add_load(sink_pin);
  clock->set_clock_source(io_driver);
  clock->set_clock_source_net(source_net);
  clock->add_inst(buf_inst);
  clock->add_net(leaf_net);
  clock->add_load(sink_pin);

  auto* idb_io_pin = idb_design.get_io_pin_list()->add_pin_list("clk_port");
  ASSERT_NE(idb_io_pin, nullptr);
  idb_io_pin->set_as_io();
  auto* io_term = idb_io_pin->set_term();
  ASSERT_NE(io_term, nullptr);
  io_term->set_name("clk_port");
  io_term->set_direction(idb::IdbConnectDirection::kInput);
  io_term->set_type(idb::IdbConnectType::kClock);
  idb_io_pin->set_average_coordinate(0, 0);

  const auto result = icts_test::runtime::CurrentRuntime().wrapper.writeClocksDetailed(
      icts_test::runtime::CurrentRuntime().design, icts_test::runtime::CurrentRuntime().reporter, {clock});

  EXPECT_TRUE(result.success);
  auto* idb_buf = idb_design.get_instance_list()->find_instance("cts_buf0");
  ASSERT_NE(idb_buf, nullptr);
  EXPECT_EQ(idb_buf->get_cell_master()->get_name(), "CTS_BUF");
  auto* root_net = idb_design.get_net_list()->find_net("root_clk_net");
  auto* leaf_idb_net = idb_design.get_net_list()->find_net("leaf_clk_net");
  ASSERT_NE(root_net, nullptr);
  ASSERT_NE(leaf_idb_net, nullptr);
  EXPECT_EQ(idb_io_pin->get_net(), root_net);
  EXPECT_EQ(idb_buf->get_pin_by_term("A")->get_net(), root_net);
  EXPECT_EQ(idb_buf->get_pin_by_term("Y")->get_net(), leaf_idb_net);
  EXPECT_EQ(idb_sink_pin->get_net(), leaf_idb_net);
}

TEST(FlowTest, WrapperClockTreeMaterializationDoesNotCreateNonCtsInstWhenResolvingClockSinkPin)
{
  ScopedFlowReset scoped_flow_reset;
  idb::IdbLayout idb_layout;
  idb::IdbDesign idb_design(&idb_layout);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_design(&idb_design);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_layout(&idb_layout);

  auto* clock = icts_test::runtime::CurrentRuntime().design.makeClock("LOGICAL_CLK", "root_clk_net");
  ASSERT_NE(clock, nullptr);
  BuildClockForWrapperClockTreeMaterialization(*clock, "clk_port", "missing_sink", "CLK");
  auto* idb_io_pin = idb_design.get_io_pin_list()->add_pin_list("clk_port");
  ASSERT_NE(idb_io_pin, nullptr);
  idb_io_pin->set_as_io();
  auto* io_term = idb_io_pin->set_term();
  ASSERT_NE(io_term, nullptr);
  io_term->set_name("clk_port");
  io_term->set_direction(idb::IdbConnectDirection::kInput);
  io_term->set_type(idb::IdbConnectType::kClock);
  idb_io_pin->set_average_coordinate(0, 0);

  const auto result = icts_test::runtime::CurrentRuntime().wrapper.writeClocksDetailed(
      icts_test::runtime::CurrentRuntime().design, icts_test::runtime::CurrentRuntime().reporter, {clock});

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.idb_clock_tree_restored);
  EXPECT_EQ(idb_design.get_instance_list()->find_instance("missing_sink"), nullptr);
  EXPECT_EQ(idb_design.get_net_list()->find_net("root_clk_net"), nullptr);
}

TEST(FlowTest, WrapperClockTreeMaterializationFailureRemovesNewCtsInstAndRestoresTouchedNetPins)
{
  ScopedFlowReset scoped_flow_reset;
  idb::IdbLayout idb_layout;
  idb::IdbDesign idb_design(&idb_layout);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_design(&idb_design);
  icts_test::runtime::CurrentRuntime().wrapper.set_idb_layout(&idb_layout);

  auto* reg_master = AddIdbCellMaster(idb_layout, "REG_CELL");
  ASSERT_NE(reg_master, nullptr);
  ASSERT_NE(AddIdbTerm(*reg_master, "CLK", idb::IdbConnectDirection::kInput, idb::IdbConnectType::kClock), nullptr);
  auto* buf_master = AddIdbCellMaster(idb_layout, "CTS_BUF");
  ASSERT_NE(buf_master, nullptr);
  ASSERT_NE(AddIdbTerm(*buf_master, "A", idb::IdbConnectDirection::kInput, idb::IdbConnectType::kClock), nullptr);
  ASSERT_NE(AddIdbTerm(*buf_master, "Y", idb::IdbConnectDirection::kOutput, idb::IdbConnectType::kClock), nullptr);
  auto* old_sink_inst = AddIdbInst(idb_design, "old_sink", *reg_master, 200, 0);
  ASSERT_NE(old_sink_inst, nullptr);
  auto* old_sink_pin = old_sink_inst->get_pin_by_term("CLK");
  ASSERT_NE(old_sink_pin, nullptr);
  auto* root_net = idb_design.get_net_list()->add_net("root_clk_net", idb::IdbConnectType::kClock);
  ASSERT_NE(root_net, nullptr);
  auto* idb_io_pin = idb_design.get_io_pin_list()->add_pin_list("clk_port");
  ASSERT_NE(idb_io_pin, nullptr);
  idb_io_pin->set_as_io();
  auto* io_term = idb_io_pin->set_term();
  ASSERT_NE(io_term, nullptr);
  io_term->set_name("clk_port");
  io_term->set_direction(idb::IdbConnectDirection::kInput);
  io_term->set_type(idb::IdbConnectType::kClock);
  idb_io_pin->set_average_coordinate(0, 0);
  AttachIdbPinToNet(*root_net, *idb_io_pin);
  AttachIdbPinToNet(*root_net, *old_sink_pin);

  auto* clock = icts_test::runtime::CurrentRuntime().design.makeClock("LOGICAL_CLK", "root_clk_net");
  ASSERT_NE(clock, nullptr);
  auto* source_net = icts_test::runtime::CurrentRuntime().design.makeNet("root_clk_net");
  auto* leaf_net = icts_test::runtime::CurrentRuntime().design.makeNet("leaf_clk_net");
  ASSERT_NE(source_net, nullptr);
  ASSERT_NE(leaf_net, nullptr);
  auto* source_pin = icts_test::runtime::CurrentRuntime().design.makePin("clk_port");
  ASSERT_NE(source_pin, nullptr);
  source_pin->set_name("clk_port");
  source_pin->set_type(icts::PinType::kOut);
  source_pin->set_io(true);
  source_pin->set_net(source_net);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(source_pin));
  auto* buf_inst = MakeDesignInst("cts_buf_restore", "CTS_BUF", icts::InstType::kBuffer, icts::Point<int>(10, 0));
  ASSERT_NE(buf_inst, nullptr);
  auto* buf_in = AddOwnedLoad(*clock, source_net, *buf_inst, "A");
  auto* buf_out = icts_test::runtime::CurrentRuntime().design.makePin("Y");
  ASSERT_NE(buf_in, nullptr);
  ASSERT_NE(buf_out, nullptr);
  buf_out->set_name("Y");
  buf_out->set_type(icts::PinType::kOut);
  buf_out->set_inst(buf_inst);
  buf_out->set_net(leaf_net);
  buf_inst->insertDriverPin(buf_out);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(buf_out));
  auto* missing_sink_inst = MakeDesignInst("missing_leaf_sink", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(300, 0));
  auto* missing_sink_pin = icts_test::runtime::CurrentRuntime().design.makePin("CLK");
  ASSERT_NE(missing_sink_inst, nullptr);
  ASSERT_NE(missing_sink_pin, nullptr);
  missing_sink_pin->set_name("CLK");
  missing_sink_pin->set_type(icts::PinType::kClock);
  missing_sink_pin->set_inst(missing_sink_inst);
  missing_sink_pin->set_net(leaf_net);
  missing_sink_inst->add_pin(missing_sink_pin);
  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.indexPin(missing_sink_pin));

  source_net->set_driver(source_pin);
  source_net->add_load(buf_in);
  leaf_net->set_driver(buf_out);
  leaf_net->add_load(missing_sink_pin);
  clock->set_clock_source(source_pin);
  clock->set_clock_source_net(source_net);
  clock->add_inst(buf_inst);
  clock->add_net(leaf_net);
  clock->add_load(missing_sink_pin);

  const auto result = icts_test::runtime::CurrentRuntime().wrapper.writeClocksDetailed(
      icts_test::runtime::CurrentRuntime().design, icts_test::runtime::CurrentRuntime().reporter, {clock});

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.idb_clock_tree_restored);
  EXPECT_EQ(idb_design.get_instance_list()->find_instance("cts_buf_restore"), nullptr);
  EXPECT_EQ(idb_design.get_net_list()->find_net("leaf_clk_net"), nullptr);
  EXPECT_EQ(idb_io_pin->get_net(), root_net);
  EXPECT_EQ(old_sink_pin->get_net(), root_net);
  EXPECT_EQ(root_net->get_pin_number(), 2);
}

}  // namespace
}  // namespace icts_test
