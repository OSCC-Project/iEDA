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
 * @file FlowRuntimeIsolationTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-24
 * @brief Same-process CTS runtime and flow isolation tests.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

#include "Config.hh"
#include "Design.hh"
#include "Flow.hh"
#include "Schema.hh"
#include "common/CTSTestRuntime.hh"
#include "database/design/Inst.hh"
#include "synthesis/trace/SynthesisTrace.hh"

namespace icts_test {
namespace {

TEST(CTSTestRuntime, LocalRuntimeFlowPairsDoNotShareCTSState)
{
  runtime::CTSTestRuntime first;
  runtime::CTSTestRuntime second;
  auto& first_runtime = *first._runtime;
  auto& second_runtime = *second._runtime;

  EXPECT_NE(&first_runtime, &second_runtime);
  ASSERT_NE(first.flow, nullptr);
  ASSERT_NE(second.flow, nullptr);
  EXPECT_NE(first.flow.get(), second.flow.get());
  EXPECT_NE(&first_runtime.config, &second_runtime.config);
  EXPECT_NE(&first_runtime.design, &second_runtime.design);
  EXPECT_NE(&first_runtime.wrapper, &second_runtime.wrapper);
  EXPECT_NE(&first_runtime.wrapper, &second_runtime.wrapper);
  EXPECT_NE(&first_runtime.fast_sta, &second_runtime.fast_sta);
  EXPECT_NE(&first_runtime.reporter, &second_runtime.reporter);

  first_runtime.config.set_max_fanout(4U);
  second_runtime.config.set_max_fanout(9U);
  EXPECT_EQ(first_runtime.config.get_max_fanout(), 4U);
  EXPECT_EQ(second_runtime.config.get_max_fanout(), 9U);

  auto* first_inst = first_runtime.design.makeInst("first_only");
  auto* second_inst = second_runtime.design.makeInst("second_only");
  ASSERT_NE(first_inst, nullptr);
  ASSERT_NE(second_inst, nullptr);
  first_inst->set_cell_master("FIRST_CELL");
  second_inst->set_cell_master("SECOND_CELL");
  EXPECT_EQ(first_runtime.design.findInst("first_only"), first_inst);
  EXPECT_EQ(first_runtime.design.findInst("second_only"), nullptr);
  EXPECT_EQ(second_runtime.design.findInst("first_only"), nullptr);
  EXPECT_EQ(second_runtime.design.findInst("second_only"), second_inst);

  const auto first_log_path = std::filesystem::temp_directory_path() / "icts_first_local_runtime.log";
  const auto second_log_path = std::filesystem::temp_directory_path() / "icts_second_local_runtime.log";
  first_runtime.reporter.open(first_log_path, "First Local Runtime");
  second_runtime.reporter.open(second_log_path, "Second Local Runtime");
  EXPECT_NE(first_runtime.reporter.getActivePath(), second_runtime.reporter.getActivePath());

  first.flow->runCTS();
  second.flow->setSetupReady(true);
  second.flow->runCTS();
  EXPECT_FALSE(first.flow->outputRunSummary().success);
  EXPECT_FALSE(second.flow->outputRunSummary().success);
  EXPECT_EQ(first.flow->outputRunSummary().outcome, icts::SynthesisOutcome::kFailed);
  EXPECT_EQ(second.flow->outputRunSummary().outcome, icts::SynthesisOutcome::kNoOp);
  EXPECT_EQ(second.flow->outputRunSummary().no_op_reason, "no_clocks_discovered");

  first.reset();
  EXPECT_EQ(first_runtime.config.get_max_fanout(), 32U);
  EXPECT_EQ(first_runtime.design.findInst("first_only"), nullptr);
  EXPECT_EQ(second_runtime.config.get_max_fanout(), 9U);
  EXPECT_EQ(second_runtime.design.findInst("second_only"), second_inst);

  second.reset();
}

}  // namespace
}  // namespace icts_test
