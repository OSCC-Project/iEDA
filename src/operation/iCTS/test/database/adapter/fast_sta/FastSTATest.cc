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
 * @file FastSTATest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Unit tests for CTS fast STA data, timing, power, and incremental APIs.
 */

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "FastStaDmpCeff.hh"
#include "FastStaIncremental.hh"
#include "FastStaParasitics.hh"
#include "FastStaPower.hh"
#include "FastStaTiming.hh"
#include "Flow.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "clock_sizing/FastStaClockSizingEdit.hh"
#include "clock_state/FastStaClockState.hh"
#include "common/CTSTestRuntime.hh"
#include "database/config/Config.hh"
#include "liberty/FastStaLibertyModel.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts_test {
namespace {

auto MakeAxis(icts::FastStaLibertyAxisKind kind, std::vector<double> values) -> icts::FastStaLibertyAxis
{
  return icts::FastStaLibertyAxis{.kind = kind, .values = std::move(values)};
}

auto MakeTable(icts::FastStaLibertyTableKind kind, double base) -> icts::FastStaLibertyTable
{
  return icts::FastStaLibertyTable{
      .kind = kind,
      .transition = icts::FastStaTransition::kRise,
      .axes
      = {MakeAxis(icts::FastStaLibertyAxisKind::kInputSlew, {0.0, 1.0}), MakeAxis(icts::FastStaLibertyAxisKind::kOutputLoad, {0.0, 2.0})},
      .values = {base, base + 0.2, base + 0.1, base + 0.3},
  };
}

auto MakeCell(const std::string& master, double input_cap_pf, double area_um2, double leakage_w) -> icts::FastStaLibertyCell
{
  return icts::FastStaLibertyCell{
      .cell_master = master,
      .input_port = "A",
      .output_port = "Y",
      .input_cap_pf = input_cap_pf,
      .output_cap_limit_pf = 3.0,
      .input_slew_limit_ns = 1.0,
      .area_um2 = area_um2,
      .voltage_v = 1.0,
      .leakage_power_w = leakage_w,
      .timing_arc = icts::FastStaLibertyArc{
          .from_port = "A",
          .to_port = "Y",
          .delay_tables = {MakeTable(icts::FastStaLibertyTableKind::kCellDelay, 0.10)},
          .slew_tables = {MakeTable(icts::FastStaLibertyTableKind::kOutputSlew, 0.20)},
          .internal_power_tables = {MakeTable(icts::FastStaLibertyTableKind::kInternalPower, 0.50)},
      },
  };
}

auto MakeOpenStaAlignmentTable(icts::FastStaLibertyTableKind kind, double low_slew_low_cap, double low_slew_high_cap,
                               double high_slew_low_cap, double high_slew_high_cap) -> icts::FastStaLibertyTable
{
  return icts::FastStaLibertyTable{
      .kind = kind,
      .transition = icts::FastStaTransition::kRise,
      .axes
      = {MakeAxis(icts::FastStaLibertyAxisKind::kInputSlew, {0.1, 0.5}), MakeAxis(icts::FastStaLibertyAxisKind::kOutputLoad, {0.1, 1.0})},
      .values = {low_slew_low_cap, low_slew_high_cap, high_slew_low_cap, high_slew_high_cap},
  };
}

auto MakeOpenStaAlignmentCell() -> icts::FastStaLibertyCell
{
  return icts::FastStaLibertyCell{
      .cell_master = "BUF_X1",
      .input_port = "A",
      .output_port = "Y",
      .input_cap_pf = 0.2,
      .output_cap_limit_pf = 3.0,
      .input_threshold_rise = 0.5,
      .input_threshold_fall = 0.5,
      .output_threshold_rise = 0.5,
      .output_threshold_fall = 0.5,
      .slew_lower_threshold_rise = 0.3,
      .slew_lower_threshold_fall = 0.3,
      .slew_upper_threshold_rise = 0.7,
      .slew_upper_threshold_fall = 0.7,
      .slew_derate_from_library = 1.0,
      .area_um2 = 1.0,
      .voltage_v = 1.0,
      .leakage_power_w = 0.00001,
      .timing_arc = icts::FastStaLibertyArc{
          .from_port = "A",
          .to_port = "Y",
          .delay_tables = {MakeOpenStaAlignmentTable(icts::FastStaLibertyTableKind::kCellDelay, 0.110, 0.200, 0.150, 0.240)},
          .slew_tables = {MakeOpenStaAlignmentTable(icts::FastStaLibertyTableKind::kOutputSlew, 0.210, 0.300, 0.250, 0.340)},
          .internal_power_tables = {MakeOpenStaAlignmentTable(icts::FastStaLibertyTableKind::kInternalPower, 0.510, 0.600, 0.550, 0.640)},
      },
  };
}

auto MakeNode(icts::FastStaNodeKind kind, std::string name, std::string inst_name, std::string pin_name, std::string cell_master,
              icts::FastStaPoint location, double input_cap_pf, icts::FastStaNetId incoming_net_id,
              std::vector<icts::FastStaNetId> output_net_ids) -> icts::FastStaNode
{
  icts::FastStaNode node;
  node.kind = kind;
  node.name = std::move(name);
  node.inst_name = std::move(inst_name);
  node.pin_name = std::move(pin_name);
  node.cell_master = std::move(cell_master);
  node.location = location;
  node.input_cap_pf = input_cap_pf;
  node.incoming_net_id = incoming_net_id;
  node.output_net_ids = std::move(output_net_ids);
  return node;
}

auto MakeRcNode(std::string name, double wire_cap_pf, double pin_cap_pf, double cap_pf, double elmore_delay_ns,
                icts::FastStaNodeId terminal_node_id) -> icts::FastStaRcNode
{
  icts::FastStaRcNode node;
  node.name = std::move(name);
  node.wire_cap_pf = wire_cap_pf;
  node.pin_cap_pf = pin_cap_pf;
  node.cap_pf = cap_pf;
  node.elmore_delay_ns = elmore_delay_ns;
  node.terminal_node_id = terminal_node_id;
  return node;
}

auto MakeParasitic(std::vector<icts::FastStaRcNode> rc_nodes, std::vector<icts::FastStaRcEdge> rc_edges,
                   icts::FastStaRcNodeId root_rc_node_id, icts::FastStaPiModel pi = {}, double total_cap_pf = 0.0,
                   bool pre_reduced_pi_elmore = false) -> icts::FastStaNetParasitic
{
  icts::FastStaNetParasitic parasitic;
  parasitic.rc_nodes = std::move(rc_nodes);
  parasitic.rc_edges = std::move(rc_edges);
  parasitic.root_rc_node_id = root_rc_node_id;
  parasitic.pi = pi;
  parasitic.total_cap_pf = total_cap_pf;
  parasitic.pre_reduced_pi_elmore = pre_reduced_pi_elmore;
  return parasitic;
}

auto MakeNet(std::string name, icts::FastStaNodeId driver_node_id, std::vector<icts::FastStaNodeId> load_node_ids, double max_cap_pf,
             icts::FastStaNetParasitic parasitic = {}) -> icts::FastStaNet
{
  icts::FastStaNet net;
  net.name = std::move(name);
  net.driver_node_id = driver_node_id;
  net.load_node_ids = std::move(load_node_ids);
  net.max_cap_pf = max_cap_pf;
  net.parasitic = std::move(parasitic);
  return net;
}

class ScopedRootInputSlew
{
 public:
  explicit ScopedRootInputSlew(double root_input_slew_ns)
      : _original_root_input_slew_ns(icts_test::runtime::CurrentRuntime().config.get_root_input_slew())
  {
    icts_test::runtime::CurrentRuntime().config.set_root_input_slew(root_input_slew_ns);
  }

  ~ScopedRootInputSlew() { icts_test::runtime::CurrentRuntime().config.set_root_input_slew(_original_root_input_slew_ns); }

  ScopedRootInputSlew(const ScopedRootInputSlew& rhs) = delete;
  ScopedRootInputSlew(ScopedRootInputSlew&& rhs) = delete;
  auto operator=(const ScopedRootInputSlew& rhs) -> ScopedRootInputSlew& = delete;
  auto operator=(ScopedRootInputSlew&& rhs) -> ScopedRootInputSlew& = delete;

 private:
  double _original_root_input_slew_ns = 0.0;
};

auto MakeTinyContext() -> icts::FastStaClockContext
{
  icts::FastStaClockContext context;
  context.clock_name = "clk";
  context.clock_net_name = "clk_net";
  context.clock_period_ns = 10.0;
  context.dbu_per_um = 1000;
  context.routing_layer = 1;
  context.liberty_cell_by_master["BUF_X1"] = MakeCell("BUF_X1", 0.20, 1.5, 0.01);
  context.liberty_cell_by_master["BUF_X2"] = MakeCell("BUF_X2", 0.40, 2.5, 0.02);

  context.source_node_id = 0U;
  context.nodes = {
      MakeNode(icts::FastStaNodeKind::kSource, "clk_src", "", "clk_src", "", icts::FastStaPoint{.x_dbu = 0, .y_dbu = 0}, 0.0,
               icts::kInvalidFastStaNetId, {0U}),
      MakeNode(icts::FastStaNodeKind::kBufferInput, "buf/A", "buf", "A", "BUF_X1", icts::FastStaPoint{.x_dbu = 1000, .y_dbu = 0}, 0.20, 0U,
               {}),
      MakeNode(icts::FastStaNodeKind::kBufferOutput, "buf/Y", "buf", "Y", "BUF_X1", icts::FastStaPoint{.x_dbu = 1000, .y_dbu = 0}, 0.0,
               icts::kInvalidFastStaNetId, {1U}),
      MakeNode(icts::FastStaNodeKind::kSink, "sink/CLK", "sink", "CLK", "", icts::FastStaPoint{.x_dbu = 2000, .y_dbu = 0}, 0.10, 1U, {}),
  };
  context.node_id_by_name = {{"clk_src", 0U}, {"buf/A", 1U}, {"buf/Y", 2U}, {"sink/CLK", 3U}};
  context.node_id_by_location = {{{0, 0}, 0U}, {{1000, 0}, 1U}, {{2000, 0}, 3U}};
  context.nets = {
      MakeNet("clk_net", 0U, {1U}, 3.0,
              MakeParasitic({MakeRcNode("clk_net@0", 0.0, 0.0, 0.0, 0.0, 0U), MakeRcNode("clk_net@1", 0.0, 0.20, 0.20, 0.0, 1U)},
                            {icts::FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = 100.0}}, 0U)),
      MakeNet("leaf_net", 2U, {3U}, 3.0,
              MakeParasitic({MakeRcNode("leaf_net@0", 0.0, 0.0, 0.0, 0.0, 2U), MakeRcNode("leaf_net@1", 0.0, 0.10, 0.10, 0.0, 3U)},
                            {icts::FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = 100.0}}, 0U)),
  };
  context.net_id_by_name = {{"clk_net", 0U}, {"leaf_net", 1U}};
  return context;
}

auto MakeTwoLevelContext() -> icts::FastStaClockContext
{
  icts::FastStaClockContext context;
  context.clock_name = "clk";
  context.clock_net_name = "clk_net";
  context.clock_period_ns = 10.0;
  context.liberty_cell_by_master["BUF_X1"] = MakeCell("BUF_X1", 0.20, 1.5, 0.01);
  context.liberty_cell_by_master["BUF_X2"] = MakeCell("BUF_X2", 0.40, 2.5, 0.02);

  context.source_node_id = 0U;
  context.nodes = {
      MakeNode(icts::FastStaNodeKind::kSource, "clk_src", "", "clk_src", "", {}, 0.0, icts::kInvalidFastStaNetId, {0U}),
      MakeNode(icts::FastStaNodeKind::kBufferInput, "buf1/A", "buf1", "A", "BUF_X1", {}, 0.20, 0U, {}),
      MakeNode(icts::FastStaNodeKind::kBufferOutput, "buf1/Y", "buf1", "Y", "BUF_X1", {}, 0.0, icts::kInvalidFastStaNetId, {1U}),
      MakeNode(icts::FastStaNodeKind::kBufferInput, "buf2/A", "buf2", "A", "BUF_X1", {}, 0.20, 1U, {}),
      MakeNode(icts::FastStaNodeKind::kBufferOutput, "buf2/Y", "buf2", "Y", "BUF_X1", {}, 0.0, icts::kInvalidFastStaNetId, {2U}),
      MakeNode(icts::FastStaNodeKind::kSink, "sink/CLK", "sink", "CLK", "", {}, 0.10, 2U, {}),
  };
  context.node_id_by_name = {{"clk_src", 0U}, {"buf1/A", 1U}, {"buf1/Y", 2U}, {"buf2/A", 3U}, {"buf2/Y", 4U}, {"sink/CLK", 5U}};
  context.nets = {
      MakeNet("clk_net", 0U, {1U}, 3.0,
              MakeParasitic({MakeRcNode("clk_net@0", 0.0, 0.0, 0.0, 0.0, 0U), MakeRcNode("clk_net@1", 0.0, 0.20, 0.20, 0.0, 1U)},
                            {icts::FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = 100.0}}, 0U)),
      MakeNet("mid_net", 2U, {3U}, 3.0,
              MakeParasitic({MakeRcNode("mid_net@0", 0.0, 0.0, 0.0, 0.0, 2U), MakeRcNode("mid_net@1", 0.0, 0.20, 0.20, 0.0, 3U)},
                            {icts::FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = 100.0}}, 0U)),
      MakeNet("leaf_net", 4U, {5U}, 3.0,
              MakeParasitic({MakeRcNode("leaf_net@0", 0.0, 0.0, 0.0, 0.0, 4U), MakeRcNode("leaf_net@1", 0.0, 0.10, 0.10, 0.0, 5U)},
                            {icts::FastStaRcEdge{.from = 0U, .to = 1U, .resistance_ohm = 100.0}}, 0U)),
  };
  context.net_id_by_name = {{"clk_net", 0U}, {"mid_net", 1U}, {"leaf_net", 2U}};
  return context;
}

auto MakeOpenStaAlignmentPathContext() -> icts::FastStaClockContext
{
  icts::FastStaClockContext context;
  context.clock_name = "clk";
  context.clock_net_name = "clk";
  context.clock_period_ns = 10.0;
  context.liberty_cell_by_master["BUF_X1"] = MakeOpenStaAlignmentCell();

  context.source_node_id = 0U;
  context.nodes = {
      MakeNode(icts::FastStaNodeKind::kSource, "clk", "", "clk", "", {}, 0.0, icts::kInvalidFastStaNetId, {0U}),
      MakeNode(icts::FastStaNodeKind::kBufferInput, "u_buf/A", "u_buf", "A", "BUF_X1", {}, 0.20, 0U, {}),
      MakeNode(icts::FastStaNodeKind::kBufferOutput, "u_buf/Y", "u_buf", "Y", "BUF_X1", {}, 0.0, icts::kInvalidFastStaNetId, {1U}),
      MakeNode(icts::FastStaNodeKind::kBufferInput, "u_leaf/A", "u_leaf", "A", "BUF_X1", {}, 0.20, 1U, {}),
      MakeNode(icts::FastStaNodeKind::kBufferOutput, "u_leaf/Y", "u_leaf", "Y", "BUF_X1", {}, 0.0, icts::kInvalidFastStaNetId, {2U}),
      MakeNode(icts::FastStaNodeKind::kSink, "sink", "", "sink", "", {}, 0.0, 2U, {}),
  };
  context.node_id_by_name = {{"clk", 0U}, {"u_buf/A", 1U}, {"u_buf/Y", 2U}, {"u_leaf/A", 3U}, {"u_leaf/Y", 4U}, {"sink", 5U}};
  context.nets = {
      MakeNet("clk", 0U, {1U}, 3.0),
      MakeNet("leaf", 2U, {3U}, 3.0,
              MakeParasitic({MakeRcNode("leaf@root", 0.80, 0.0, 0.80, 0.0, 2U), MakeRcNode("leaf@u_leaf/A", 0.0, 0.20, 0.20, 0.15, 3U)}, {},
                            0U, icts::FastStaPiModel{.near_cap_pf = 0.20, .resistance_ohm = 1000.0, .far_cap_pf = 0.80}, 1.0, true)),
      MakeNet("sink", 4U, {5U}, 3.0,
              MakeParasitic({MakeRcNode("sink@root", 0.50, 0.0, 0.50, 0.0, 4U), MakeRcNode("sink@load", 0.0, 0.0, 0.0, 0.08, 5U)}, {}, 0U,
                            icts::FastStaPiModel{.near_cap_pf = 0.10, .resistance_ohm = 1000.0, .far_cap_pf = 0.40}, 0.50, true)),
  };
  context.net_id_by_name = {{"clk", 0U}, {"leaf", 1U}, {"sink", 2U}};
  return context;
}

TEST(FastSTATest, LibertyTableBilinearLookupInterpolates)
{
  const auto table = MakeTable(icts::FastStaLibertyTableKind::kCellDelay, 1.0);
  const auto value = table.lookup(0.5, 1.0);

  if (!value.has_value()) {
    ADD_FAILURE() << "Expected bilinear lookup result.";
    return;
  }
  EXPECT_NEAR(*value, 1.15, 1e-12);
}

TEST(FastSTATest, DmpDriverTimingProducesCeffAndLoadSlew)
{
  const icts::FastStaPiModel pi{.near_cap_pf = 0.2, .resistance_ohm = 1000.0, .far_cap_pf = 0.8};
  const auto cell = MakeCell("BUF_X1", 0.20, 1.5, 0.01);

  const auto driver_timing = icts::FastStaDmpCeff::calcDriverTiming(cell, pi, icts::FastStaTransition::kRise, 0.2);

  ASSERT_TRUE(driver_timing.valid);
  EXPECT_GT(driver_timing.ceff_pf, 0.0);
  EXPECT_LE(driver_timing.ceff_pf, 1.0);
  EXPECT_GT(driver_timing.gate_delay_ns, 0.0);
  EXPECT_GT(driver_timing.driver_slew_ns, 0.0);

  const auto load_timing = icts::FastStaDmpCeff::calcLoadDelaySlew(driver_timing, 0.15, nullptr);
  EXPECT_TRUE(load_timing.valid);
  EXPECT_GT(load_timing.wire_delay_ns, 0.0);
  EXPECT_GE(load_timing.load_slew_ns, driver_timing.driver_slew_ns);
}

TEST(FastSTATest, DmpDriverTimingMatchesOpenStaMicroCase)
{
  const auto cell = MakeOpenStaAlignmentCell();
  const icts::FastStaPiModel pi{.near_cap_pf = 0.2, .resistance_ohm = 1000.0, .far_cap_pf = 0.8};

  const auto driver_timing = icts::FastStaDmpCeff::calcDriverTiming(cell, pi, icts::FastStaTransition::kRise, 0.2);

  ASSERT_TRUE(driver_timing.valid);
  EXPECT_NEAR(driver_timing.ceff_pf, 0.433933, 1e-6) << "actual=" << driver_timing.ceff_pf;
  EXPECT_NEAR(driver_timing.gate_delay_ns, 0.153393, 1e-6) << "actual=" << driver_timing.gate_delay_ns;
  EXPECT_NEAR(driver_timing.driver_slew_ns, 0.265526, 1e-6) << "actual=" << driver_timing.driver_slew_ns;

  const auto load_timing = icts::FastStaDmpCeff::calcLoadDelaySlew(driver_timing, 0.15, nullptr);
  ASSERT_TRUE(load_timing.valid);
  EXPECT_NEAR(load_timing.wire_delay_ns, 0.143337, 1e-6) << "actual=" << load_timing.wire_delay_ns;
  EXPECT_NEAR(load_timing.load_slew_ns, 0.278977, 1e-6) << "actual=" << load_timing.load_slew_ns;
}

TEST(FastSTATest, TimingPropagationMatchesOpenStaTwoLevelPath)
{
  auto context = MakeOpenStaAlignmentPathContext();
  context.root_input_slew_ns = 0.2;
  ASSERT_TRUE(icts::FastStaTiming::update(context));

  ASSERT_TRUE(context.nodes.at(1U).timing.valid);
  ASSERT_TRUE(context.nodes.at(2U).timing.valid);
  ASSERT_TRUE(context.nodes.at(3U).timing.valid);
  ASSERT_TRUE(context.nodes.at(4U).timing.valid);
  ASSERT_TRUE(context.nodes.at(5U).timing.valid);

  EXPECT_NEAR(context.nodes.at(1U).timing.arrival_ns, 0.0, 1e-9);
  EXPECT_NEAR(context.nodes.at(1U).timing.slew_ns, 0.199999988, 1e-6);
  EXPECT_NEAR(context.nodes.at(2U).timing.arrival_ns, 0.153393298, 1e-6);
  EXPECT_NEAR(context.nodes.at(2U).timing.slew_ns, 0.265526026, 1e-6);
  EXPECT_NEAR(context.nodes.at(3U).timing.arrival_ns, 0.296730995, 1e-6);
  EXPECT_NEAR(context.nodes.at(3U).timing.slew_ns, 0.278976977, 1e-6);
  EXPECT_NEAR(context.nodes.at(4U).timing.arrival_ns, 0.443657875, 1e-6);
  EXPECT_NEAR(context.nodes.at(4U).timing.slew_ns, 0.254852772, 1e-6);
  EXPECT_NEAR(context.nodes.at(5U).timing.arrival_ns, 0.522887707, 1e-6);
  EXPECT_NEAR(context.nodes.at(5U).timing.slew_ns, 0.257477403, 1e-6);
}

TEST(FastSTATest, TimingUsesContextRootSlewWithoutConfigMutation)
{
  const ScopedRootInputSlew root_input_slew_guard(0.777);

  auto fast_context = MakeOpenStaAlignmentPathContext();
  fast_context.root_input_slew_ns = 0.12;
  ASSERT_TRUE(icts::FastStaTiming::update(fast_context));

  auto slow_context = MakeOpenStaAlignmentPathContext();
  slow_context.root_input_slew_ns = 0.48;
  ASSERT_TRUE(icts::FastStaTiming::update(slow_context));

  EXPECT_NEAR(fast_context.nodes.at(1U).timing.slew_ns, 0.12, 1e-12);
  EXPECT_NEAR(slow_context.nodes.at(1U).timing.slew_ns, 0.48, 1e-12);
  EXPECT_NEAR(icts_test::runtime::CurrentRuntime().config.get_root_input_slew(), 0.777, 1e-12);
}

TEST(FastSTATest, SourceBoundaryNetUsesNormalNetLoadAndCapFields)
{
  auto context = MakeTinyContext();
  context.nets.at(0U).max_cap_pf = 0.15;

  icts::FastStaParasitics::updateNetLoads(context);

  const auto& source_boundary_net = context.nets.at(0U);
  EXPECT_EQ(source_boundary_net.driver_node_id, context.source_node_id);
  EXPECT_NEAR(source_boundary_net.load_cap_pf, 0.20, 1e-12);
  EXPECT_NEAR(source_boundary_net.max_cap_pf, 0.15, 1e-12);
  EXPECT_GT(source_boundary_net.load_cap_pf, source_boundary_net.max_cap_pf);
}

TEST(FastSTATest, PiElmoreReductionPropagatesDownstreamCapAndElmore)
{
  auto context = MakeTinyContext();
  icts::FastStaParasitics::updateNetLoads(context);

  ASSERT_TRUE(icts::FastStaParasitics::reduceToPiElmore(context, 0U));

  const auto& parasitic = context.nets.at(0U).parasitic;
  EXPECT_TRUE(parasitic.valid);
  EXPECT_NEAR(parasitic.total_cap_pf, 0.20, 1e-12);
  EXPECT_NEAR(parasitic.rc_nodes.at(1U).downstream_cap_pf, 0.20, 1e-12);
  EXPECT_NEAR(parasitic.rc_nodes.at(1U).elmore_delay_ns, 0.02, 1e-12);
}

TEST(FastSTATest, TimingPowerAndMasterChangeUpdateContext)
{
  auto context = MakeTinyContext();

  ASSERT_TRUE(icts::FastStaTiming::update(context));
  EXPECT_TRUE(context.skew.valid);
  EXPECT_EQ(context.skew.max_sink_name, "sink/CLK");
  EXPECT_GT(context.nodes.at(3U).timing.arrival_ns, 0.0);
  EXPECT_TRUE(icts::FastStaPower::update(context));
  EXPECT_NEAR(context.power.area_um2, 1.5, 1e-12);
  EXPECT_GT(context.power.switching_power_w, 0.0);
  EXPECT_GT(context.power.internal_power_w, 0.0);
  EXPECT_NEAR(context.power.leakage_power_w, 0.01, 1e-12);

  ASSERT_TRUE(icts::FastStaIncremental::changeBufferMaster(context, 1U, "BUF_X2"));
  EXPECT_EQ(context.nodes.at(1U).cell_master, "BUF_X2");
  EXPECT_EQ(context.nodes.at(2U).cell_master, "BUF_X2");
  EXPECT_NEAR(context.nodes.at(1U).input_cap_pf, 0.40, 1e-12);
  ASSERT_TRUE(icts::FastStaTiming::update(context));
  ASSERT_TRUE(icts::FastStaPower::update(context));
  EXPECT_NEAR(context.power.area_um2, 2.5, 1e-12);
  EXPECT_NEAR(context.power.leakage_power_w, 0.02, 1e-12);
}

TEST(FastSTATest, IncrementalMasterChangeMatchesFullRecompute)
{
  auto incremental_context = MakeTwoLevelContext();
  ASSERT_TRUE(icts::FastStaTiming::update(incremental_context));
  ASSERT_TRUE(icts::FastStaPower::update(incremental_context));

  auto full_context = incremental_context;

  const auto dirty_region_opt = icts::FastStaIncremental::changeBufferMasterIncremental(incremental_context, 3U, "BUF_X2");
  if (!dirty_region_opt.has_value()) {
    ADD_FAILURE() << "Expected incremental dirty region.";
    return;
  }
  const auto& dirty_region = *dirty_region_opt;
  ASSERT_TRUE(dirty_region.valid);
  EXPECT_EQ(dirty_region.start_node_id, 1U);
  EXPECT_FALSE(dirty_region.net_ids.empty());

  ASSERT_TRUE(icts::FastStaTiming::updateRegion(incremental_context, dirty_region));
  ASSERT_TRUE(icts::FastStaPower::updateRegion(incremental_context, dirty_region));

  ASSERT_TRUE(icts::FastStaIncremental::changeBufferMaster(full_context, 3U, "BUF_X2"));
  ASSERT_TRUE(icts::FastStaTiming::update(full_context));
  ASSERT_TRUE(icts::FastStaPower::update(full_context));

  ASSERT_TRUE(incremental_context.skew.valid);
  ASSERT_TRUE(full_context.skew.valid);
  EXPECT_NEAR(incremental_context.nodes.at(3U).input_cap_pf, full_context.nodes.at(3U).input_cap_pf, 1e-12);
  EXPECT_NEAR(incremental_context.nets.at(1U).load_cap_pf, full_context.nets.at(1U).load_cap_pf, 1e-12);
  EXPECT_NEAR(incremental_context.nodes.at(5U).timing.arrival_ns, full_context.nodes.at(5U).timing.arrival_ns, 1e-12);
  EXPECT_NEAR(incremental_context.nodes.at(5U).timing.slew_ns, full_context.nodes.at(5U).timing.slew_ns, 1e-12);
  EXPECT_NEAR(incremental_context.power.switching_power_w, full_context.power.switching_power_w, 1e-18);
  EXPECT_NEAR(incremental_context.power.internal_power_w, full_context.power.internal_power_w, 1e-18);
  EXPECT_NEAR(incremental_context.power.leakage_power_w, full_context.power.leakage_power_w, 1e-18);
  EXPECT_NEAR(incremental_context.power.area_um2, full_context.power.area_um2, 1e-12);
}

}  // namespace
}  // namespace icts_test
