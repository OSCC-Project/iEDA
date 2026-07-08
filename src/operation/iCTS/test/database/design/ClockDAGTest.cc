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
 * @file ClockDAGTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-06
 * @brief Unit tests for Design-owned CTS clock DAG path-depth semantics
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Flow.hh"
#include "common/CTSTestRuntime.hh"
#include "database/design/Clock.hh"
#include "database/design/ClockDAG.hh"
#include "database/design/Design.hh"
#include "database/design/Inst.hh"
#include "database/design/Net.hh"
#include "database/design/Pin.hh"
#include "database/io/Wrapper.hh"
#include "database/spatial/Point.hh"
#include "flow/evaluation/qor/QorEvaluation.hh"

namespace icts_test {
namespace {

class ScopedDesignReset
{
 public:
  ScopedDesignReset()
  {
    icts_test::runtime::CurrentRuntime().wrapper.reset();
    icts_test::runtime::CurrentRuntime().design.reset();
  }
  ~ScopedDesignReset()
  {
    icts_test::runtime::CurrentRuntime().wrapper.reset();
    icts_test::runtime::CurrentRuntime().design.reset();
  }
};

struct BufferPins
{
  icts::Inst* inst = nullptr;
  icts::Pin* input = nullptr;
  icts::Pin* output = nullptr;
};

struct ClockPins
{
  icts::Clock* clock = nullptr;
  icts::Pin* source = nullptr;
  icts::Net* source_net = nullptr;
};

auto makeInst(const std::string& name, icts::InstType type, const icts::Point<int>& location) -> icts::Inst*
{
  auto* inst = icts_test::runtime::CurrentRuntime().design.makeInst(name);
  inst->set_name(name);
  inst->set_cell_master(type == icts::InstType::kBuffer ? "BUF_X1" : "REG_X1");
  inst->set_type(type);
  inst->set_location(location);
  return inst;
}

auto makePin(const std::string& name, icts::PinType type, icts::Inst* inst, const icts::Point<int>& location) -> icts::Pin*
{
  auto* pin = icts_test::runtime::CurrentRuntime().design.makePin(name);
  pin->set_name(name);
  pin->set_type(type);
  pin->set_inst(inst);
  pin->set_location(location);
  pin->set_io(false);
  if (inst != nullptr) {
    if (type == icts::PinType::kOut) {
      inst->insertDriverPin(pin);
    } else {
      inst->add_pin(pin);
    }
  }
  (void) icts_test::runtime::CurrentRuntime().design.indexPin(pin);
  return pin;
}

auto connectNet(const std::string& name, icts::Pin* driver, const std::vector<icts::Pin*>& loads) -> icts::Net*
{
  auto* net = icts_test::runtime::CurrentRuntime().design.makeNet(name);
  net->set_name(name);
  net->set_driver(driver);
  net->set_loads({});
  if (driver != nullptr) {
    driver->set_net(net);
  }
  for (auto* load : loads) {
    net->add_load(load);
    if (load != nullptr) {
      load->set_net(net);
    }
  }
  return net;
}

auto makeEvaluationInput() -> icts::EvaluationInput
{
  auto& shared = icts_test::runtime::CurrentRuntime();
  return icts::EvaluationInput{
      .config = &shared.config,
      .clock_layout = nullptr,
      .design = &shared.design,
      .wrapper = &shared.wrapper,
      .reporter = &shared.reporter,
  };
}

auto makeClock(const std::string& clock_name, const std::string& net_name) -> ClockPins
{
  auto* clock = icts_test::runtime::CurrentRuntime().design.makeClock(clock_name, net_name);
  clock->set_clock_name(clock_name);
  clock->set_clock_net_name(net_name);
  auto* source = makePin(clock_name + "_src", icts::PinType::kOut, nullptr, icts::Point<int>(0, 0));
  auto* source_net = connectNet(net_name, source, {});
  clock->set_clock_source(source);
  clock->set_clock_source_net(source_net);
  return ClockPins{.clock = clock, .source = source, .source_net = source_net};
}

auto makeBuffer(const std::string& name, int x) -> BufferPins
{
  auto* inst = makeInst(name, icts::InstType::kBuffer, icts::Point<int>(x, 0));
  inst->set_cell_master("");
  auto* output = makePin("Y", icts::PinType::kOut, inst, inst->get_location());
  auto* input = makePin("A", icts::PinType::kIn, inst, inst->get_location());
  return BufferPins{.inst = inst, .input = input, .output = output};
}

auto makeSink(const std::string& name, icts::InstType type, int x) -> icts::Pin*
{
  auto* inst = makeInst(name, type, icts::Point<int>(x, 0));
  return makePin("CLK", icts::PinType::kClock, inst, inst->get_location());
}

TEST(DesignIndexTest, RemoveClockMembershipObjectsErasesRecordedInstAndNetNames)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buffer = makeBuffer("buffer_to_remove", 10);
  auto* sink = makeSink("sink", icts::InstType::kFlipFlop, 20);
  auto* removed_net = connectNet("net_to_remove", buffer.output, {sink});
  clock_pins.clock->add_inst(buffer.inst);
  clock_pins.clock->add_net(removed_net);
  clock_pins.clock->add_load(sink);

  buffer.inst->set_name("renamed_buffer_to_remove");
  removed_net->set_name("renamed_net_to_remove");

  auto& design = icts_test::runtime::CurrentRuntime().design;
  design.removeClockMembershipObjects(*clock_pins.clock);

  EXPECT_EQ(design.findInst("buffer_to_remove"), nullptr);
  EXPECT_EQ(design.findInst("renamed_buffer_to_remove"), nullptr);
  EXPECT_EQ(design.findNet("net_to_remove"), nullptr);
  EXPECT_EQ(design.findNet("renamed_net_to_remove"), nullptr);
}

TEST(ClockDAGTest, BranchPathsReportSourceToFlipFlopBufferDepths)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buf_one = makeBuffer("buf_one", 10);
  auto buf_two_a = makeBuffer("buf_two_a", 20);
  auto buf_two_b = makeBuffer("buf_two_b", 30);
  auto* ff_one = makeSink("ff_one", icts::InstType::kFlipFlop, 40);
  auto* ff_two = makeSink("ff_two", icts::InstType::kFlipFlop, 50);

  connectNet("clk_net", clock_pins.source, {buf_one.input, buf_two_a.input});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* one_deep_net = connectNet("one_deep_net", buf_one.output, {ff_one});
  auto* two_deep_mid_net = connectNet("two_deep_mid_net", buf_two_a.output, {buf_two_b.input});
  auto* two_deep_leaf_net = connectNet("two_deep_leaf_net", buf_two_b.output, {ff_two});

  clock_pins.clock->add_inst(buf_one.inst);
  clock_pins.clock->add_inst(buf_two_a.inst);
  clock_pins.clock->add_inst(buf_two_b.inst);
  clock_pins.clock->add_net(one_deep_net);
  clock_pins.clock->add_net(two_deep_mid_net);
  clock_pins.clock->add_net(two_deep_leaf_net);
  clock_pins.clock->add_load(ff_one);
  clock_pins.clock->add_load(ff_two);

  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  const auto stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_pins.clock);
  EXPECT_TRUE(stats.available);
  EXPECT_EQ(stats.status, "available");
  EXPECT_EQ(stats.min_buffer_count, 1);
  EXPECT_EQ(stats.max_buffer_count, 2);
  EXPECT_EQ(stats.ff_sink_terminal_count, 2U);
}

TEST(ClockDAGTest, DirectSourceToFlipFlopPathReportsZeroBuffers)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto* ff = makeSink("ff_zero", icts::InstType::kFlipFlop, 10);
  connectNet("clk_net", clock_pins.source, {ff});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  clock_pins.clock->add_load(ff);

  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  const auto stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_pins.clock);
  EXPECT_TRUE(stats.available);
  EXPECT_EQ(stats.min_buffer_count, 0);
  EXPECT_EQ(stats.max_buffer_count, 0);
}

TEST(ClockDAGTest, BoundaryLoadDoesNotRequireBufferInputArc)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto* boundary_load = makeSink("comb_boundary", icts::InstType::kBoundaryLoad, 10);
  auto* latch_sink = makeSink("latch_sink", icts::InstType::kLatch, 20);
  connectNet("clk_net", clock_pins.source, {boundary_load, latch_sink});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  clock_pins.clock->add_load(boundary_load);
  clock_pins.clock->add_load(latch_sink);

  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  const auto stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_pins.clock);
  EXPECT_TRUE(stats.available);
  EXPECT_EQ(stats.status, "available");
  EXPECT_EQ(stats.min_buffer_count, 0);
  EXPECT_EQ(stats.max_buffer_count, 0);
  EXPECT_EQ(stats.ff_sink_terminal_count, 1U);
}

TEST(ClockDAGTest, MalformedTrueBufferInvalidatesTopology)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto* buffer_inst = makeInst("malformed_buffer", icts::InstType::kBuffer, icts::Point<int>(10, 0));
  auto* buffer_output = makePin("Y", icts::PinType::kOut, buffer_inst, buffer_inst->get_location());
  auto* latch_sink = makeSink("latch_sink", icts::InstType::kLatch, 20);
  connectNet("clk_net", clock_pins.source, {buffer_output});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* leaf_net = connectNet("leaf_net", buffer_output, {latch_sink});
  clock_pins.clock->add_inst(buffer_inst);
  clock_pins.clock->add_net(leaf_net);
  clock_pins.clock->add_load(latch_sink);

  EXPECT_FALSE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  EXPECT_NE(icts_test::runtime::CurrentRuntime().design.get_clock_dag().get_status().find("clock_cell_input_pin_is_null"),
            std::string::npos);
}

TEST(ClockDAGTest, NoFlipFlopTerminalIsUnavailableAndDoesNotReuseTotalBufferCount)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buffer = makeBuffer("buf_no_ff", 10);
  auto* macro_sink = makeSink("macro_sink", icts::InstType::kMacroBlock, 20);
  connectNet("clk_net", clock_pins.source, {buffer.input});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* leaf_net = connectNet("macro_leaf_net", buffer.output, {macro_sink});

  clock_pins.clock->add_inst(buffer.inst);
  clock_pins.clock->add_net(leaf_net);
  clock_pins.clock->add_load(macro_sink);

  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  const auto stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_pins.clock);
  EXPECT_FALSE(stats.available);
  EXPECT_EQ(stats.status, "no_ff_sink_terminal");
  EXPECT_EQ(stats.min_buffer_count, 0);
  EXPECT_EQ(stats.max_buffer_count, 0);
}

TEST(ClockDAGTest, CycleInvalidatesTopologyAndPathStats)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buffer = makeBuffer("buf_loop", 10);
  connectNet("clk_net", clock_pins.source, {buffer.input});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* loop_net = connectNet("loop_net", buffer.output, {buffer.input});
  clock_pins.clock->add_inst(buffer.inst);
  clock_pins.clock->add_net(loop_net);

  EXPECT_FALSE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  EXPECT_TRUE(icts_test::runtime::CurrentRuntime().design.get_clock_dag().hasCycle(clock_pins.clock));
  const auto stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_pins.clock);
  EXPECT_FALSE(stats.available);
  EXPECT_EQ(stats.status, "invalid_topology");
}

TEST(ClockDAGTest, MultiClockQueriesRemainIsolated)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_zero = makeClock("clk_zero", "clk_zero_net");
  auto* ff_zero = makeSink("ff_zero", icts::InstType::kFlipFlop, 10);
  connectNet("clk_zero_net", clock_zero.source, {ff_zero});
  clock_zero.clock->set_clock_source_net(clock_zero.source_net);
  clock_zero.clock->add_load(ff_zero);

  auto clock_two = makeClock("clk_two", "clk_two_net");
  auto buf_a = makeBuffer("clk_two_buf_a", 20);
  auto buf_b = makeBuffer("clk_two_buf_b", 30);
  auto* ff_two = makeSink("ff_two", icts::InstType::kFlipFlop, 40);
  connectNet("clk_two_net", clock_two.source, {buf_a.input});
  clock_two.clock->set_clock_source_net(clock_two.source_net);
  auto* mid_net = connectNet("clk_two_mid_net", buf_a.output, {buf_b.input});
  auto* leaf_net = connectNet("clk_two_leaf_net", buf_b.output, {ff_two});
  clock_two.clock->add_inst(buf_a.inst);
  clock_two.clock->add_inst(buf_b.inst);
  clock_two.clock->add_net(mid_net);
  clock_two.clock->add_net(leaf_net);
  clock_two.clock->add_load(ff_two);

  ASSERT_TRUE(icts_test::runtime::CurrentRuntime().design.rebuildClockDAG());
  const auto zero_stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_zero.clock);
  EXPECT_TRUE(zero_stats.available);
  EXPECT_EQ(zero_stats.min_buffer_count, 0);
  EXPECT_EQ(zero_stats.max_buffer_count, 0);

  const auto two_stats = icts_test::runtime::CurrentRuntime().design.get_clock_dag().pathBufferStats(clock_two.clock);
  EXPECT_TRUE(two_stats.available);
  EXPECT_EQ(two_stats.min_buffer_count, 2);
  EXPECT_EQ(two_stats.max_buffer_count, 2);
}

TEST(ClockDAGTest, QorEvaluationPathDepthFieldsUseSourceToFlipFlopDAGStats)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buf_one = makeBuffer("qor_buf_one", 10);
  auto buf_two_a = makeBuffer("qor_buf_two_a", 20);
  auto buf_two_b = makeBuffer("qor_buf_two_b", 30);
  auto* ff_one = makeSink("qor_ff_one", icts::InstType::kFlipFlop, 40);
  auto* ff_two = makeSink("qor_ff_two", icts::InstType::kFlipFlop, 50);

  connectNet("clk_net", clock_pins.source, {buf_one.input, buf_two_a.input});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* one_deep_net = connectNet("qor_one_deep_net", buf_one.output, {ff_one});
  auto* two_deep_mid_net = connectNet("qor_two_deep_mid_net", buf_two_a.output, {buf_two_b.input});
  auto* two_deep_leaf_net = connectNet("qor_two_deep_leaf_net", buf_two_b.output, {ff_two});

  clock_pins.clock->add_inst(buf_one.inst);
  clock_pins.clock->add_inst(buf_two_a.inst);
  clock_pins.clock->add_inst(buf_two_b.inst);
  clock_pins.clock->add_net(one_deep_net);
  clock_pins.clock->add_net(two_deep_mid_net);
  clock_pins.clock->add_net(two_deep_leaf_net);
  clock_pins.clock->add_load(ff_one);
  clock_pins.clock->add_load(ff_two);

  icts::EvaluationState state;
  icts::QorEvaluation::evaluate(state, makeEvaluationInput());
  const auto summary = icts::QorEvaluation::outputSummary(state);

  EXPECT_EQ(summary.final_clock_buffer_count, 3);
  EXPECT_EQ(summary.clock_member_buffer_count, 3);
  EXPECT_EQ(summary.buffer_num, 3);
  EXPECT_EQ(summary.path_depth_metric_status, "available");
  EXPECT_EQ(summary.clock_path_min_buffer, 1);
  EXPECT_EQ(summary.clock_path_max_buffer, 2);
  EXPECT_EQ(summary.feature_max_clock_network_level, 2);
}

TEST(ClockDAGTest, QorEvaluationNoFlipFlopPathDepthIsUnavailableZeroNotTotalBuffers)
{
  const ScopedDesignReset scoped_design_reset;

  auto clock_pins = makeClock("clk", "clk_net");
  auto buffer = makeBuffer("qor_buf_no_ff", 10);
  auto* macro_sink = makeSink("qor_macro_sink", icts::InstType::kMacroBlock, 20);
  connectNet("clk_net", clock_pins.source, {buffer.input});
  clock_pins.clock->set_clock_source_net(clock_pins.source_net);
  auto* leaf_net = connectNet("qor_macro_leaf_net", buffer.output, {macro_sink});

  clock_pins.clock->add_inst(buffer.inst);
  clock_pins.clock->add_net(leaf_net);
  clock_pins.clock->add_load(macro_sink);

  icts::EvaluationState state;
  icts::QorEvaluation::evaluate(state, makeEvaluationInput());
  const auto summary = icts::QorEvaluation::outputSummary(state);

  EXPECT_EQ(summary.final_clock_buffer_count, 1);
  EXPECT_EQ(summary.clock_member_buffer_count, 1);
  EXPECT_EQ(summary.buffer_num, 1);
  EXPECT_EQ(summary.path_depth_metric_status, "no_ff_sink_terminal");
  EXPECT_EQ(summary.clock_path_min_buffer, 0);
  EXPECT_EQ(summary.clock_path_max_buffer, 0);
  EXPECT_EQ(summary.feature_max_clock_network_level, 0);
}

}  // namespace
}  // namespace icts_test
