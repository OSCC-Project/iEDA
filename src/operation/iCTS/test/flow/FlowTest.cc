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
 * @file FlowTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-25
 * @brief Lightweight interface and sink-domain wiring tests for Flow.
 */

#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "CTSAPI.hh"
#include "CTSStatus.hh"
#include "Clock.hh"
#include "ClockLayout.hh"
#include "Config.hh"
#include "Design.hh"
#include "Flow.hh"
#include "FlowDesignFixture.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"
#include "Point.hh"
#include "Schema.hh"
#include "common/CTSTestRuntime.hh"
#include "common/logging/LogText.hh"
#include "evaluation/qor/QorEvaluation.hh"
#include "feature_icts.h"
#include "synthesis/Synthesis.hh"
#include "synthesis/distribution/ClockDistribution.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/realization/ClockTreeRealization.hh"
#include "synthesis/topology/Topology.hh"
#include "synthesis/trace/SynthesisTrace.hh"
#include "synthesis/trace/domain_status/DomainStatus.hh"

namespace icts_test {
namespace {

using namespace flow_test;

TEST(FlowTest, EmptyFlowRunIsCallable)
{
  ScopedFlowReset scoped_flow_reset;

  scoped_flow_reset.flow.setSetupReady(true);
  const auto status = scoped_flow_reset.flow.runCTS();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code, icts::FlowRunStatusCode::kNoOp);

  const auto summary = scoped_flow_reset.flow.outputRunSummary();
  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.outcome, icts::SynthesisOutcome::kNoOp);
  EXPECT_EQ(summary.no_op_reason, "no_clocks_discovered");
  EXPECT_EQ(summary.total_clocks, 0U);
  EXPECT_EQ(summary.successful_clocks, 0U);
  EXPECT_EQ(summary.skipped_clocks, 0U);
  EXPECT_EQ(summary.failed_clocks, 0U);
}

TEST(FlowTest, RunWithoutSetupFailsExplicitlyAndSkipsPipelineStages)
{
  ScopedFlowReset scoped_flow_reset;

  const auto cts_log_path = icts_test::runtime::CurrentRuntime().reporter.getActivePath();
  ASSERT_FALSE(cts_log_path.empty());
  const auto status = scoped_flow_reset.flow.runCTS();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code, icts::FlowRunStatusCode::kSetupNotReady);
  EXPECT_EQ(status.step, "setup");

  const auto cts_log_content = ReadTextFile(cts_log_path);
  ASSERT_FALSE(cts_log_content.empty());
  EXPECT_EQ(cts_log_content.find("## Input Overview"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("## Synthesis Overview"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("### Synthesis Flow"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("## Evaluation Overview"), std::string::npos);
  EXPECT_NE(cts_log_content.find("## Runtime Overview"), std::string::npos);
  EXPECT_NE(cts_log_content.find("## Run Results"), std::string::npos);
  EXPECT_NE(cts_log_content.find("CTS Runtime Overview"), std::string::npos);
  EXPECT_NE(cts_log_content.find("CTS Key Results"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("CTS Evaluation Overview"), std::string::npos);
  EXPECT_NE(cts_log_content.find("setup_failed"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Notes"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Outcome"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("outcome"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Runtime rows are consolidated here"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("This section collects the final user-facing outcome"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Evaluation writes the final CTS clock tree back to iDB"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("memory_delta_mb"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("elapsed_time_s"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("final_buffer_area_um2"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("max_clock_net_wirelength_um"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("total_clock_network_wirelength_um"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("max_clock_net_wirelength_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("total_clock_network_wirelength_dbu"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("design_dbu_per_um"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("unique_clock_domains"), std::string::npos);
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*max_clock_net_wirelength\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_TRUE(std::regex_search(cts_log_content, std::regex(R"(\|\s*total_clock_network_wirelength\s*\|\s*[^|\n]*um\s*\|)")));
  EXPECT_EQ(cts_log_content.find("clock_path_min_buffer"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("clock_path_max_buffer"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("feature_max_clock_network_level"), std::string::npos);

  const auto runtime_read_data_summary = common::logging::ExtractTextBlock(cts_log_content, "ReadData Overview");
  const auto read_data_summary = common::logging::ExtractTextBlock(cts_log_content, "CTSReadData Read CTS clock data Overview");
  const auto flow_summary = common::logging::ExtractTextBlock(cts_log_content, "CTSFlow Run CTS synthesis flow Overview");
  const auto evaluation_summary = common::logging::ExtractTextBlock(cts_log_content, "Evaluation Evaluate CTS clock tree Overview");
  const auto main_flow_summary = common::logging::ExtractTextBlock(cts_log_content, "CTS Clock tree synthesis API flow Summary");
  EXPECT_TRUE(runtime_read_data_summary.empty());
  EXPECT_TRUE(read_data_summary.empty());
  EXPECT_TRUE(flow_summary.empty());
  EXPECT_TRUE(evaluation_summary.empty());
  ASSERT_FALSE(main_flow_summary.empty());
  EXPECT_NE(main_flow_summary.find("setup_failed"), std::string::npos);
  EXPECT_NE(main_flow_summary.find("status"), std::string::npos);
  EXPECT_EQ(main_flow_summary.find("main_flow"), std::string::npos);
  EXPECT_EQ(main_flow_summary.find("outcome"), std::string::npos);
  EXPECT_EQ(main_flow_summary.find("elapsed_time"), std::string::npos);
}

TEST(FlowTest, AllSkippedSynthesisReportsNoOp)
{
  ScopedFlowReset scoped_flow_reset;

  auto pins = AddClockToDesign(nullptr, nullptr);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.clock_source, nullptr);
  ASSERT_NE(pins.clock_net, nullptr);
  ASSERT_TRUE(pins.clock->get_loads().empty());

  icts::ClockLayout clock_layout;
  icts::CharacterizationLibrary char_library;
  auto& runtime = icts_test::runtime::CurrentRuntime();
  const auto summary = icts::Synthesis::run(icts::SynthesisInput{
      .config = &runtime.config,
      .design = &runtime.design,
      .wrapper = &runtime.wrapper,
      .fast_sta = &runtime.fast_sta,
      .reporter = &runtime.reporter,
      .clock_layout = &clock_layout,
      .characterization_library = &char_library,
  });

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.outcome, icts::SynthesisOutcome::kNoOp);
  EXPECT_EQ(summary.no_op_reason, "all_clocks_skipped");
  EXPECT_EQ(summary.total_clocks, 1U);
  EXPECT_EQ(summary.successful_clocks, 0U);
  EXPECT_EQ(summary.skipped_clocks, 1U);
  EXPECT_EQ(summary.failed_clocks, 0U);
  EXPECT_TRUE(DomainStatusContains(summary.domain_status, "skipped", "none", "no valid sinks"));
  EXPECT_TRUE(clock_layout.get_clocks().empty());
}

TEST(FlowTest, NoOpRunDoesNotExposeFinalEvaluationSummary)
{
  ScopedFlowReset scoped_flow_reset;

  scoped_flow_reset.flow.setSetupReady(true);
  const auto status = scoped_flow_reset.flow.runCTS();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code, icts::FlowRunStatusCode::kNoOp);

  const auto run_summary = scoped_flow_reset.flow.outputRunSummary();
  EXPECT_EQ(run_summary.outcome, icts::SynthesisOutcome::kNoOp);

  const auto qor_summary = scoped_flow_reset.flow.outputSummary();
  EXPECT_FALSE(qor_summary.has_evaluation_result);
  EXPECT_EQ(qor_summary.qor_metric_status, "unavailable");
  EXPECT_EQ(qor_summary.physical_metric_source, "unavailable");
}

TEST(CTSAPITest, InitAndRunExposeConfigFailureStatus)
{
  const auto work_dir = std::filesystem::temp_directory_path() / "icts_api_missing_config_status";
  const auto missing_config = work_dir / "missing_cts_config.json";

  const auto setup_status = CTS_API_INST.init(missing_config.string(), work_dir.string());
  EXPECT_FALSE(setup_status.ok());
  EXPECT_EQ(setup_status.code, icts::CTSStatusCode::kConfigError);
  EXPECT_NE(setup_status.message.find("failed to open iCTS config file"), std::string::npos);

  const auto run_status = CTS_API_INST.runCTS();
  EXPECT_FALSE(run_status.ok());
  EXPECT_EQ(run_status.code, icts::CTSStatusCode::kNotInitialized);

  CTS_API_INST.resetAPI();
}

TEST(FlowTest, ClockDistributionSummaryUsesMacroSinkTerminology)
{
  ScopedFlowReset scoped_flow_reset;
  auto& shared = icts_test::runtime::CurrentRuntime();

  const auto cts_log_path = shared.reporter.getActivePath();
  ASSERT_FALSE(cts_log_path.empty());

  auto* macro_inst = MakeDesignInst("macro0", "MACRO_CELL", icts::InstType::kMacroBlock, icts::Point<int>(100, 0));
  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(200, 0));
  auto pins = AddClockToDesign(macro_inst, reg_inst);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.macro_sink, nullptr);
  ASSERT_NE(pins.regular_sink, nullptr);

  shared.design.emitClockDistributionSummary(shared.reporter);

  const auto cts_log_content = ReadTextFile(cts_log_path);
  ASSERT_FALSE(cts_log_content.empty());
  const auto distribution_summary = common::logging::ExtractTextBlock(cts_log_content, "Clock Distribution Overview");
  ASSERT_FALSE(distribution_summary.empty());
  EXPECT_NE(distribution_summary.find("Macro Sinks"), std::string::npos);
  EXPECT_EQ(distribution_summary.find("Buffer Sinks"), std::string::npos);
  EXPECT_TRUE(std::regex_search(distribution_summary, std::regex(R"(\|\s*clk\s*\|\s*1\s*\|\s*2\s*\|\s*1\s*\|\s*1\s*\|\s*0\s*\|)")));
}

TEST(FlowTest, MixedMacroAndRegularSingleSinkDomainsUseSeparateDownstreamNets)
{
  ScopedFlowReset scoped_flow_reset;

  auto* macro_inst = MakeDesignInst("macro0", "MACRO_CELL", icts::InstType::kMacroBlock, icts::Point<int>(100, 0));
  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(200, 0));
  auto pins = AddClockToDesign(macro_inst, reg_inst);

  PrepareDirectRootBufferNets(*pins.clock, "CTS_TEST_BUF", "A", "Y");

  ASSERT_NE(pins.clock, nullptr);
  auto* macro_root = FindInstByNamePart(pins.clock->get_insts(), "hard_macro_root_buf");
  auto* regular_root = FindInstByNamePart(pins.clock->get_insts(), "regular_root_buf");
  ASSERT_NE(macro_root, nullptr);
  ASSERT_NE(regular_root, nullptr);
  auto* macro_root_input = FindInputPin(macro_root);
  auto* regular_root_input = FindInputPin(regular_root);
  auto* macro_root_output = macro_root->findDriverPin();
  auto* regular_root_output = regular_root->findDriverPin();
  ASSERT_NE(macro_root_input, nullptr);
  ASSERT_NE(regular_root_input, nullptr);
  ASSERT_NE(macro_root_output, nullptr);
  ASSERT_NE(regular_root_output, nullptr);

  EXPECT_EQ(pins.clock->get_clock_source_net(), pins.clock_net);
  EXPECT_EQ(pins.clock_net->get_driver(), pins.clock_source);
  EXPECT_EQ(pins.clock_source->get_net(), pins.clock_net);
  EXPECT_EQ(pins.clock_net->get_loads().size(), 2U);
  EXPECT_TRUE(ContainsPin(pins.clock_net->get_loads(), macro_root_input));
  EXPECT_TRUE(ContainsPin(pins.clock_net->get_loads(), regular_root_input));

  auto* macro_sink_net = pins.macro_sink->get_net();
  auto* regular_sink_net = pins.regular_sink->get_net();
  ASSERT_NE(macro_sink_net, nullptr);
  ASSERT_NE(regular_sink_net, nullptr);
  EXPECT_NE(macro_sink_net, regular_sink_net);
  EXPECT_EQ(macro_sink_net, FindNetByNamePart(pins.clock->get_nets(), "hard_macro_downstream_net"));
  EXPECT_EQ(regular_sink_net, FindNetByNamePart(pins.clock->get_nets(), "regular_downstream_net"));
  EXPECT_EQ(macro_sink_net->get_driver(), macro_root_output);
  EXPECT_EQ(regular_sink_net->get_driver(), regular_root_output);
  ASSERT_EQ(macro_sink_net->get_loads().size(), 1U);
  ASSERT_EQ(regular_sink_net->get_loads().size(), 1U);
  EXPECT_EQ(macro_sink_net->get_loads().front(), pins.macro_sink);
  EXPECT_EQ(regular_sink_net->get_loads().front(), pins.regular_sink);
}

TEST(FlowTest, RepeatedNetPreparationRestoresClockSourceNetBeforeRebuildingInsertedNets)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);

  PrepareDirectRootBufferNets(*pins.clock, "CTS_TEST_BUF", "A", "Y");
  ASSERT_NE(pins.clock, nullptr);
  EXPECT_NE(pins.regular_sink->get_net(), pins.clock_net);
  ASSERT_EQ(pins.clock->get_nets().size(), 1U);
  const auto first_design_inst_count = icts_test::runtime::CurrentRuntime().design.get_insts().size();
  const auto first_design_pin_count = icts_test::runtime::CurrentRuntime().design.get_pins().size();
  const auto first_design_net_count = icts_test::runtime::CurrentRuntime().design.get_nets().size();

  PrepareDirectRootBufferNets(*pins.clock, "CTS_TEST_BUF", "A", "Y");
  EXPECT_EQ(icts_test::runtime::CurrentRuntime().design.get_insts().size(), first_design_inst_count);
  EXPECT_EQ(icts_test::runtime::CurrentRuntime().design.get_pins().size(), first_design_pin_count);
  EXPECT_EQ(icts_test::runtime::CurrentRuntime().design.get_nets().size(), first_design_net_count);
  ASSERT_EQ(pins.clock->get_insts().size(), 1U);
  ASSERT_EQ(pins.clock->get_nets().size(), 1U);
  auto* root_buffer = pins.clock->get_insts().front();
  auto* root_input = FindInputPin(root_buffer);
  ASSERT_NE(root_input, nullptr);

  ASSERT_NE(pins.clock->get_clock_source_net(), nullptr);
  EXPECT_EQ(pins.clock->get_clock_source_net(), pins.clock_net);
  EXPECT_EQ(pins.clock_net->get_driver(), pins.clock_source);
  ASSERT_EQ(pins.clock_net->get_loads().size(), 1U);
  EXPECT_EQ(pins.clock_net->get_loads().front(), root_input);
  ASSERT_NE(pins.regular_sink->get_net(), nullptr);
  EXPECT_EQ(pins.regular_sink->get_net(), pins.clock->get_nets().front());
  EXPECT_EQ(pins.regular_sink->get_net()->get_loads().front(), pins.regular_sink);
}

TEST(FlowTest, RootBufferInsertionFailureRestoresClockMembershipAndRecordsSinkDomainStatus)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.regular_sink, nullptr);

  icts::ClockLayout clock_layout;
  icts::CharacterizationLibrary char_library;
  auto& runtime = icts_test::runtime::CurrentRuntime();
  const auto summary = icts::Synthesis::run(icts::SynthesisInput{
      .config = &runtime.config,
      .design = &runtime.design,
      .wrapper = &runtime.wrapper,
      .fast_sta = &runtime.fast_sta,
      .reporter = &runtime.reporter,
      .clock_layout = &clock_layout,
      .characterization_library = &char_library,
  });

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.failed_clocks, 1U);
  EXPECT_EQ(summary.skipped_clocks, 0U);
  EXPECT_TRUE(DomainStatusContains(summary.domain_status, "failed", "regular", "failed to insert root buffer"));
  ExpectClockRestoredToOriginalLoads(pins);
  EXPECT_TRUE(clock_layout.get_clocks().empty());
}

TEST(FlowTest, DownstreamNetCreationFailureRestoresClockMembershipAndRecordsSinkDomainStatus)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.regular_sink, nullptr);

  const auto domain_prefix = icts::ClockTreeRealization::makeSinkDomainPrefix(*pins.clock, 0U, icts::SinkDomainKind::kRegular);
  auto* existing_downstream_net = icts_test::runtime::CurrentRuntime().design.makeNet(domain_prefix + "_downstream_net");
  ASSERT_NE(existing_downstream_net, nullptr);

  icts::TableRows rows;
  icts::DomainStatusTable status_table(rows);
  const auto root_spec = MakeRootBufferSpec();
  const auto prepared = icts::ClockDistribution::prepare(icts::ClockDistributionInput{
      .design = &icts_test::runtime::CurrentRuntime().design,
      .clock = pins.clock,
      .wrapper = &icts_test::runtime::CurrentRuntime().wrapper,
      .clock_index = 0U,
      .sink_domain = icts::SinkDomainKind::kRegular,
      .sinks = std::vector<icts::Pin*>{pins.regular_sink},
      .valid_sinks = 1U,
      .root_buffer_types = icts_test::runtime::CurrentRuntime().config.get_buffer_types(),
      .status_table = &status_table,
      .root_buffer_spec = &root_spec,
  });

  EXPECT_FALSE(prepared.has_value());
  EXPECT_TRUE(StatusRowsContain(rows, "failed", "regular", "failed to create downstream net"));
  icts::Topology::resetClockTopology(*pins.clock);
  ExpectClockRestoredToOriginalLoads(pins);
}

TEST(FlowTest, TopologyResetRestoresPreparedSinkDomainAndKeepsPendingClockLayoutUnmerged)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.regular_sink, nullptr);

  icts::TableRows rows;
  icts::DomainStatusTable status_table(rows);
  const auto root_spec = MakeRootBufferSpec();
  const auto context = icts::ClockDistribution::prepare(icts::ClockDistributionInput{
      .design = &icts_test::runtime::CurrentRuntime().design,
      .clock = pins.clock,
      .wrapper = &icts_test::runtime::CurrentRuntime().wrapper,
      .clock_index = 0U,
      .sink_domain = icts::SinkDomainKind::kRegular,
      .sinks = std::vector<icts::Pin*>{pins.regular_sink},
      .valid_sinks = 1U,
      .root_buffer_types = icts_test::runtime::CurrentRuntime().config.get_buffer_types(),
      .status_table = &status_table,
      .root_buffer_spec = &root_spec,
  });
  ASSERT_TRUE(context.has_value());

  icts::ClockLayout clock_layout;
  icts::Topology::resetClockTopology(*pins.clock);
  ExpectClockRestoredToOriginalLoads(pins);
  EXPECT_TRUE(clock_layout.get_clocks().empty());
}

TEST(FlowTest, SourceToRootFailureRestoresPreparedSinkDomainsAndRecordsStatus)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);
  ASSERT_NE(pins.clock, nullptr);
  ASSERT_NE(pins.regular_sink, nullptr);

  icts::TableRows rows;
  icts::DomainStatusTable status_table(rows);
  const auto root_spec = MakeRootBufferSpec();
  auto context = icts::ClockDistribution::prepare(icts::ClockDistributionInput{
      .design = &icts_test::runtime::CurrentRuntime().design,
      .clock = pins.clock,
      .wrapper = &icts_test::runtime::CurrentRuntime().wrapper,
      .clock_index = 0U,
      .sink_domain = icts::SinkDomainKind::kRegular,
      .sinks = std::vector<icts::Pin*>{pins.regular_sink},
      .valid_sinks = 1U,
      .root_buffer_types = icts_test::runtime::CurrentRuntime().config.get_buffer_types(),
      .status_table = &status_table,
      .root_buffer_spec = &root_spec,
  });
  if (!context.has_value()) {
    GTEST_FAIL() << std::string("Clock distribution context should be prepared.");
  }
  auto prepared_context = context.value();
  ASSERT_FALSE(pins.clock->get_insts().empty());
  ASSERT_FALSE(pins.clock->get_nets().empty());

  icts::ClockLayout clock_layout;
  icts::SynthesisTraceSummary summary;
  icts::CharacterizationLibrary char_library;
  prepared_context.root_input = nullptr;
  std::vector<icts::ClockDistributionContext> sink_domains = {prepared_context};

  auto& runtime = icts_test::runtime::CurrentRuntime();
  EXPECT_FALSE(icts::Topology::formClock(icts::ClockTopologyInput{
      .config = &runtime.config,
      .design = &runtime.design,
      .wrapper = &runtime.wrapper,
      .fast_sta = &runtime.fast_sta,
      .reporter = &runtime.reporter,
      .clock = pins.clock,
      .clock_index = 0U,
      .clock_layout = &clock_layout,
      .summary = &summary,
      .status_table = &status_table,
      .characterization_library = &char_library,
      .valid_sinks = 1U,
      .sink_domains = &sink_domains,
  }));
  EXPECT_TRUE(StatusRowsContain(rows, "failed", "source_to_root", "empty_root_inputs"));
  ExpectClockRestoredToOriginalLoads(pins);
}

TEST(FlowTest, EvaluateWithoutSuccessfulInstantiationLeavesSummaryUnavailableAndResetKeepsItCleared)
{
  ScopedFlowReset scoped_flow_reset;

  auto* reg_inst = MakeDesignInst("reg0", "REG_CELL", icts::InstType::kFlipFlop, icts::Point<int>(100, 0));
  auto pins = AddClockToDesign(nullptr, reg_inst);
  PrepareDirectRootBufferNets(*pins.clock, "CTS_TEST_BUF", "A", "Y");

  EXPECT_FALSE(scoped_flow_reset.flow.outputSummary().has_evaluation_result);
  EXPECT_EQ(scoped_flow_reset.flow.outputSummary().final_clock_buffer_count, 0);
  EXPECT_EQ(icts::CTSAPI::outputSummary().buffer_num, 0);

  icts::CTSAPI::resetAPI();
  const auto summary = icts::CTSAPI::outputSummary();
  EXPECT_EQ(summary.buffer_num, 0);
  EXPECT_EQ(summary.clock_path_min_buffer, 0);
}

}  // namespace
}  // namespace icts_test
