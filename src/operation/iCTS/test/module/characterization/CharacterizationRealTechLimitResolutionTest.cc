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
 * @file CharacterizationRealTechLimitResolutionTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Asset-dependent limit resolution and table-axis coverage on real-tech assets.
 */

#include <gtest/gtest.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "BufferingPattern.hh"
#include "Point.hh"
#include "SegmentChar.hh"
#include "characterization/Characterization.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/logging/LogText.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/design/Design.hh"
#include "database/design/Inst.hh"
#include "database/design/Pin.hh"
#include "database/io/Wrapper.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"

namespace icts_test {
namespace {

namespace realtech_fixture = characterization::realtech;

auto ReadTextFile(const std::filesystem::path& path) -> std::string
{
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return {};
  }

  std::ostringstream content_stream;
  content_stream << input_stream.rdbuf();
  return content_stream.str();
}

auto ReadSchemaFieldValue(const std::string& content, const std::string& field) -> std::optional<std::string>
{
  const std::regex row_regex(R"(\|\s*)" + field + R"(\s*\|\s*([^|\n]+?)\s*\|)");
  std::smatch match;
  if (!std::regex_search(content, match, row_regex) || match.size() < 2U) {
    return std::nullopt;
  }
  return match[1].str();
}

auto ReadSchemaUnsignedFieldValue(const std::string& content, const std::string& field) -> std::optional<unsigned long long>
{
  const auto value = ReadSchemaFieldValue(content, field);
  if (!value.has_value()) {
    return std::nullopt;
  }

  try {
    return std::stoull(*value);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

TEST(CharacterizationRealTechLimitResolutionTest, UsesStrongestBufferHeightWhenWirelengthUnitMissing)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("auto_wirelength_unit", std::nullopt, 0.0, 0.0, true); prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  const auto buffer_cells = realtech_fixture::CollectConfiguredBufferLimitInfo();
  const auto usable_buffers = realtech_fixture::CollectUsableBufferMasters(buffer_cells);
  if (usable_buffers.empty()) {
    GTEST_SKIP() << "No configured buffer has both drive-cap data and physical height in real-tech assets.";
  }

  const double expected_unit_um = realtech_fixture::ResolveDefaultWirelengthUnitUm(buffer_cells, usable_buffers);
  ASSERT_GT(expected_unit_um, 0.0);

  icts::CharBuilder builder;
  const auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
  builder.init(contract.input, contract.config);
  EXPECT_DOUBLE_EQ(builder.get_wirelength_unit_um(), expected_unit_um);
  EXPECT_EQ(builder.get_wirelength_iterations(), realtech_fixture::kRealTechCharWirelengthIterations);

  const auto cts_log_path = common::io::ResolveOutputDir() / "characterization" / "realtech" / "auto_wirelength_unit" / "cts.log";
  const auto cts_log_content = ReadTextFile(cts_log_path);
  ASSERT_FALSE(cts_log_content.empty());
  const auto first_line_break = cts_log_content.find('\n');
  ASSERT_NE(first_line_break, std::string::npos);
  EXPECT_EQ(cts_log_content.find("Generate the report at "), first_line_break + 1U);
  EXPECT_NE(cts_log_content.find("CharBuilder Setup"), std::string::npos);
  const auto char_builder_setup = common::logging::ExtractTextBlock(cts_log_content, "CharBuilder Setup");
  ASSERT_FALSE(char_builder_setup.empty());
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*routing_layer\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*wire_width\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*max_slew\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*max_cap\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*wirelength_iterations\s*\|)")));
  EXPECT_TRUE(
      std::regex_search(char_builder_setup, std::regex(R"(\|\s*resolved_wirelength_unit\s*\|\s*[^|\n]*um\s*\|\s*auto_derived\s*\|)")));
  EXPECT_TRUE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*wirelength_points\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*slew_steps\s*\|)")));
  EXPECT_FALSE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*cap_steps\s*\|)")));
  EXPECT_EQ(cts_log_content.find("CharBuilder Runtime Configuration"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("CharBuilder Initialization Parameters"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("CharBuilder Routing / Wire RC"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Notes"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Characterization setup lists the resolved limits"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("wirelength_setup_source"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("deduplicated"), std::string::npos);
  EXPECT_NE(cts_log_content.find("auto_derived"), std::string::npos);
  EXPECT_NE(cts_log_content.find("strongest buffer"), std::string::npos);
  EXPECT_TRUE(std::regex_search(char_builder_setup, std::regex(R"(\|\s*routing_rc\s*\|\s*Runtime Routing / Wire RC\s*\|)")));
}

TEST(CharacterizationRealTechLimitResolutionTest, RepresentativePinCapRemainsStableThroughWrapperQuery)
{
  const auto& setup_state = common::realtech::EnsureRealTechSetup();
  if (setup_state.mode != common::realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    GTEST_SKIP() << setup_state.summary;
    return;
  }

  const auto buffer_cells = realtech_fixture::CollectConfiguredBufferLimitInfo();
  ASSERT_FALSE(buffer_cells.empty());

  const auto probe = common::realtech::TryFindRepresentativeRealPinCapProbe();
  if (!probe.has_value()) {
    GTEST_SKIP() << "Cannot find a representative real-design load pin with resolvable capacitance.";
    return;
  }

  EXPECT_GT(probe->pre_timing_cap_pf, 0.0);

  icts::Inst probe_inst(probe->inst_name, probe->cell_master, icts::InstType::kUnknown, icts::Point<int>(-1, -1));
  icts::Pin probe_pin(probe->pin_name, icts::PinType::kIn, icts::Point<int>(-1, -1), &probe_inst);

  ASSERT_FALSE(icts_test::runtime::CurrentRuntime().design.get_clocks().empty())
      << "Real-tech setup should materialize at least one SDC-declared clock.";

  const double wrapper_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryPinCapacitance(&probe_pin);
  EXPECT_GT(wrapper_cap_pf, 0.0);
  const double cap_tolerance_pf = probe->pre_timing_cap_pf * 1e-6 + 1e-6;
  EXPECT_NEAR(wrapper_cap_pf, probe->pre_timing_cap_pf, cap_tolerance_pf);

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(6);
  report_stream << "scenario=wrapper_pin_cap_probe\n";
  report_stream << "net_name=" << probe->net_name << "\n";
  report_stream << "is_clock_net=" << (probe->is_clock_net ? "true" : "false") << "\n";
  report_stream << "inst_name=" << probe->inst_name << "\n";
  report_stream << "cell_master=" << probe->cell_master << "\n";
  report_stream << "pin_name=" << probe->pin_name << "\n";
  report_stream << "initial_pin_cap_pf=" << probe->pre_timing_cap_pf << "\n";
  report_stream << "wrapper_cap_pf=" << wrapper_cap_pf << "\n";
  report_stream << "clock_count=" << icts_test::runtime::CurrentRuntime().design.get_clocks().size() << "\n";
  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("wrapper_pin_cap_probe", "wrapper_pin_cap_probe_report.txt", report_stream.str()));
}

TEST(CharacterizationRealTechLimitResolutionTest, TableAxisLimitsMatchAvailableAssetCoverage)
{
  std::vector<realtech_fixture::BufferLimitInfo> buffer_cells;
  {
    realtech_fixture::RealTechCharFixture baseline_fixture;
    if (const auto prepare_error = baseline_fixture.prepare("limit_inventory", std::nullopt, 0.0, 0.0); prepare_error.has_value()) {
      GTEST_SKIP() << *prepare_error;
      return;
    }

    buffer_cells = realtech_fixture::CollectConfiguredBufferLimitInfo();
  }

  if (buffer_cells.empty()) {
    GTEST_SKIP() << "No configured buffers with resolvable input and output pins in real-tech assets.";
  }

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(6);
  report_stream << "scenario=table_axis_limit_resolution\n";
  report_stream << "buffer_inventory_begin\n";
  for (const auto& info : buffer_cells) {
    report_stream << "buffer{cell=" << info.cell_master << ",input_pin=" << info.input_pin << ",output_pin=" << info.output_pin
                  << ",port_slew_limit_ns=" << info.port_slew_limit_ns << ",table_slew_limit_ns=" << info.table_slew_limit_ns
                  << ",port_cap_limit_pf=" << info.port_cap_limit_pf << ",table_cap_limit_pf=" << info.table_cap_limit_pf << "}\n";
  }
  report_stream << "buffer_inventory_end\n";

  bool exercised = false;
  std::vector<std::string> limitation_notes;

  const auto cap_table_only_buffers = realtech_fixture::CollectMastersByPredicate(buffer_cells, [](const auto& info) -> bool {
    const bool has_cap_table_only = info.port_cap_limit_pf <= 0.0 && info.table_cap_limit_pf > 0.0;
    const bool has_any_slew = info.port_slew_limit_ns > 0.0 || info.table_slew_limit_ns > 0.0;
    return has_cap_table_only && has_any_slew;
  });
  if (!cap_table_only_buffers.empty()) {
    realtech_fixture::RealTechCharFixture cap_fixture;
    const auto prepare_error = cap_fixture.prepare("cap_table_axis_limit", cap_table_only_buffers, 0.1, 0.0);
    ASSERT_FALSE(prepare_error.has_value()) << (prepare_error.has_value() ? *prepare_error : "");

    icts::CharBuilder builder;
    const auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
    builder.init(contract.input, contract.config);
    const double expected_cap = realtech_fixture::MinPositiveResolvedLimit(buffer_cells, cap_table_only_buffers, false);
    ASSERT_GT(expected_cap, 0.0);
    EXPECT_DOUBLE_EQ(builder.get_max_cap(), expected_cap);
    report_stream << "cap_table_only_buffers=" << realtech_fixture::JoinStrings(cap_table_only_buffers) << "\n";
    report_stream << "cap_table_only_expected_max_cap_pf=" << expected_cap << "\n";
    exercised = true;
  } else {
    limitation_notes.emplace_back("cap_table_only_buffers_unavailable");
  }

  const auto slew_table_only_buffers = realtech_fixture::CollectMastersByPredicate(buffer_cells, [](const auto& info) -> bool {
    const bool has_slew_table_only = info.port_slew_limit_ns <= 0.0 && info.table_slew_limit_ns > 0.0;
    const bool has_any_cap = info.port_cap_limit_pf > 0.0 || info.table_cap_limit_pf > 0.0;
    return has_slew_table_only && has_any_cap;
  });
  if (!slew_table_only_buffers.empty()) {
    realtech_fixture::RealTechCharFixture slew_fixture;
    const auto prepare_error = slew_fixture.prepare("slew_table_axis_limit", slew_table_only_buffers, 0.0, 1.0);
    ASSERT_FALSE(prepare_error.has_value()) << (prepare_error.has_value() ? *prepare_error : "");

    icts::CharBuilder builder;
    const auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
    builder.init(contract.input, contract.config);
    const double expected_slew = realtech_fixture::MinPositiveResolvedLimit(buffer_cells, slew_table_only_buffers, true);
    ASSERT_GT(expected_slew, 0.0);
    EXPECT_DOUBLE_EQ(builder.get_max_slew(), expected_slew);
    report_stream << "slew_table_only_buffers=" << realtech_fixture::JoinStrings(slew_table_only_buffers) << "\n";
    report_stream << "slew_table_only_expected_max_slew_ns=" << expected_slew << "\n";
    exercised = true;
  } else {
    limitation_notes.emplace_back("slew_table_only_buffers_unavailable");
  }

  report_stream << "table_axis_limit_exercised=" << (exercised ? "true" : "false") << "\n";
  report_stream << "limitations=" << realtech_fixture::JoinStrings(limitation_notes) << "\n";
  ASSERT_TRUE(
      realtech_fixture::WriteScenarioLog("table_axis_limit_resolution", "table_axis_limit_resolution_report.txt", report_stream.str()));

  if (!exercised) {
    GTEST_SKIP() << "Current real-tech assets cannot directly exercise table-axis-only limit resolution. "
                 << realtech_fixture::JoinStrings(limitation_notes);
  }
}

TEST(CharacterizationRealTechLimitResolutionTest, OverflowSamplesAreSkippedAndReportedWithinLatticeBounds)
{
  std::vector<realtech_fixture::BufferLimitInfo> buffer_cells;
  std::vector<std::string> usable_buffers;
  double baseline_max_slew = 0.0;
  double baseline_max_cap = 0.0;

  {
    realtech_fixture::RealTechCharFixture baseline_fixture;
    if (const auto prepare_error = baseline_fixture.prepare("overflow_skip_baseline", std::nullopt, 0.0, 0.0); prepare_error.has_value()) {
      GTEST_SKIP() << *prepare_error;
      return;
    }

    buffer_cells = realtech_fixture::CollectConfiguredBufferLimitInfo();
    usable_buffers = realtech_fixture::CollectUsableBufferMasters(buffer_cells);
    if (usable_buffers.empty()) {
      GTEST_SKIP() << "No configured buffer has both slew and cap limits via port or table limits.";
      return;
    }

    baseline_max_slew = realtech_fixture::MinPositiveResolvedLimit(buffer_cells, usable_buffers, true);
    baseline_max_cap = realtech_fixture::MinPositiveResolvedLimit(buffer_cells, usable_buffers, false);
  }

  ASSERT_GT(baseline_max_slew, 0.0);
  ASSERT_GT(baseline_max_cap, 0.0);
  const double constrained_max_slew = baseline_max_slew * 0.5;
  const double constrained_max_cap = baseline_max_cap * 0.5;
  ASSERT_GT(constrained_max_slew, 0.0);
  ASSERT_GT(constrained_max_cap, 0.0);

  realtech_fixture::RealTechCharFixture overflow_fixture;
  const auto prepare_error = overflow_fixture.prepare("overflow_skip_reporting", std::nullopt, constrained_max_slew, constrained_max_cap);
  ASSERT_FALSE(prepare_error.has_value()) << (prepare_error.has_value() ? *prepare_error : "");

  icts::CharBuilder builder;
  const auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
  builder.init(contract.input, contract.config);
  EXPECT_DOUBLE_EQ(builder.get_max_slew(), constrained_max_slew);
  EXPECT_DOUBLE_EQ(builder.get_max_cap(), constrained_max_cap);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  const auto lattice_summary = realtech_fixture::SummarizeSegmentCharLattice(builder.get_segment_chars(), builder);
  EXPECT_EQ(lattice_summary.out_of_range_entries, 0U) << realtech_fixture::FormatSegmentCharLatticeSummary(lattice_summary, builder);
  EXPECT_LE(lattice_summary.max_length_idx, builder.get_wirelength_iterations());
  EXPECT_LE(lattice_summary.max_input_slew_idx, builder.get_slew_steps());
  EXPECT_LE(lattice_summary.max_output_slew_idx, builder.get_slew_steps());
  EXPECT_LE(lattice_summary.max_driven_cap_idx, builder.get_cap_steps());
  EXPECT_LE(lattice_summary.max_load_cap_idx, builder.get_cap_steps());

  const bool saw_output_overflow = builder.get_output_slew_overflow_samples() > 0U;
  const bool saw_driven_cap_overflow = builder.get_driven_cap_overflow_samples() > 0U;
  EXPECT_TRUE(saw_output_overflow || saw_driven_cap_overflow);
  if (saw_output_overflow) {
    EXPECT_GT(builder.get_max_observed_output_slew_idx(), builder.get_slew_steps());
  }
  if (saw_driven_cap_overflow) {
    EXPECT_GT(builder.get_max_observed_driven_cap_idx(), builder.get_cap_steps());
  }

  const auto cts_log_path = common::io::ResolveOutputDir() / "characterization" / "realtech" / "overflow_skip_reporting" / "cts.log";
  const auto cts_log_content = ReadTextFile(cts_log_path);
  ASSERT_FALSE(cts_log_content.empty());
  EXPECT_NE(cts_log_content.find("CharBuilder Results"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Notes"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("Characterization results summarize generated entries"), std::string::npos);
  EXPECT_NE(cts_log_content.find("executed_sta_samples"), std::string::npos);
  EXPECT_NE(cts_log_content.find("output_slew_overflow_samples"), std::string::npos);
  EXPECT_NE(cts_log_content.find("output_slew_overflow_ratio"), std::string::npos);
  EXPECT_NE(cts_log_content.find("driven_cap_overflow_samples"), std::string::npos);
  EXPECT_NE(cts_log_content.find("driven_cap_overflow_ratio"), std::string::npos);
  EXPECT_NE(cts_log_content.find("max_observed_output_slew_idx"), std::string::npos);
  EXPECT_NE(cts_log_content.find("max_observed_driven_cap_idx"), std::string::npos);
  EXPECT_EQ(cts_log_content.find("max_configured_slew"), std::string::npos);

  const auto logged_output_overflow_samples = ReadSchemaUnsignedFieldValue(cts_log_content, "output_slew_overflow_samples");
  const auto logged_driven_cap_overflow_samples = ReadSchemaUnsignedFieldValue(cts_log_content, "driven_cap_overflow_samples");
  const auto logged_max_output_slew_idx = ReadSchemaUnsignedFieldValue(cts_log_content, "max_observed_output_slew_idx");
  const auto logged_max_driven_cap_idx = ReadSchemaUnsignedFieldValue(cts_log_content, "max_observed_driven_cap_idx");
  ASSERT_TRUE(logged_output_overflow_samples.has_value());
  ASSERT_TRUE(logged_driven_cap_overflow_samples.has_value());
  ASSERT_TRUE(logged_max_output_slew_idx.has_value());
  ASSERT_TRUE(logged_max_driven_cap_idx.has_value());
  const auto logged_output_overflow_samples_value = logged_output_overflow_samples.value_or(0ULL);
  const auto logged_driven_cap_overflow_samples_value = logged_driven_cap_overflow_samples.value_or(0ULL);
  const auto logged_max_output_slew_idx_value = logged_max_output_slew_idx.value_or(0ULL);
  const auto logged_max_driven_cap_idx_value = logged_max_driven_cap_idx.value_or(0ULL);
  EXPECT_EQ(logged_output_overflow_samples_value, builder.get_output_slew_overflow_samples());
  EXPECT_EQ(logged_driven_cap_overflow_samples_value, builder.get_driven_cap_overflow_samples());
  EXPECT_EQ(logged_max_output_slew_idx_value, builder.get_max_observed_output_slew_idx());
  EXPECT_EQ(logged_max_driven_cap_idx_value, builder.get_max_observed_driven_cap_idx());
}

TEST(CharacterizationRealTechLimitResolutionTest, RepeatedReducedBuildsRemainUsableWithinOnePreparedFixture)
{
  const auto& setup_state = common::realtech::EnsureRealTechSetup();
  if (setup_state.mode != common::realtech::RealTechMode::kRealTech || !setup_state.setup_succeeded) {
    GTEST_SKIP() << setup_state.summary;
    return;
  }

  const auto buffer_cells = realtech_fixture::CollectConfiguredBufferLimitInfo();
  const auto usable_buffers = realtech_fixture::CollectUsableBufferMasters(buffer_cells);
  if (usable_buffers.empty()) {
    GTEST_SKIP() << "No configured buffer has both slew and cap limits via port or table limits.";
    return;
  }

  realtech_fixture::RealTechCharFixture char_fixture;
  const auto prepare_error = char_fixture.prepare("repeat_reduced_builds", std::vector<std::string>{usable_buffers.front()}, 0.0, 0.0);
  if (prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  auto reduced_contract = realtech_fixture::MakeRuntimeCharBuilderContract();
  reduced_contract.config.wirelength_iterations = 2U;

  icts::CharBuilder first_builder;
  first_builder.init(reduced_contract.input, reduced_contract.config);
  first_builder.build();
  ASSERT_FALSE(first_builder.get_segment_chars().empty());
  const auto first_summary = realtech_fixture::SummarizeSegmentCharLattice(first_builder.get_segment_chars(), first_builder);
  EXPECT_EQ(first_summary.out_of_range_entries, 0U) << realtech_fixture::FormatSegmentCharLatticeSummary(first_summary, first_builder);

  icts::CharBuilder second_builder;
  second_builder.init(reduced_contract.input, reduced_contract.config);
  second_builder.build();
  ASSERT_FALSE(second_builder.get_segment_chars().empty());
  const auto second_summary = realtech_fixture::SummarizeSegmentCharLattice(second_builder.get_segment_chars(), second_builder);
  EXPECT_EQ(second_summary.out_of_range_entries, 0U) << realtech_fixture::FormatSegmentCharLatticeSummary(second_summary, second_builder);

  EXPECT_EQ(first_builder.get_wirelength_iterations(), reduced_contract.config.wirelength_iterations.value_or(0U));
  EXPECT_EQ(second_builder.get_wirelength_iterations(), reduced_contract.config.wirelength_iterations.value_or(0U));
  EXPECT_EQ(first_builder.get_segment_chars().size(), second_builder.get_segment_chars().size());
  EXPECT_EQ(first_builder.get_buffering_patterns().size(), second_builder.get_buffering_patterns().size());
  EXPECT_EQ(first_summary.total_entries, second_summary.total_entries);
  EXPECT_EQ(first_summary.max_length_idx, second_summary.max_length_idx);
  EXPECT_EQ(first_summary.max_input_slew_idx, second_summary.max_input_slew_idx);
  EXPECT_FALSE(icts_test::runtime::CurrentRuntime().design.get_clocks().empty())
      << "Clock data should remain materialized after repeated char-only builds.";
  EXPECT_EQ(first_summary.max_output_slew_idx, second_summary.max_output_slew_idx);
  EXPECT_EQ(first_summary.max_driven_cap_idx, second_summary.max_driven_cap_idx);
  EXPECT_EQ(first_summary.max_load_cap_idx, second_summary.max_load_cap_idx);

  std::ostringstream report_stream;
  report_stream << "scenario=repeat_reduced_builds\n";
  report_stream << "selected_buffer=" << usable_buffers.front() << "\n";
  report_stream << "wirelength_iterations=" << reduced_contract.config.wirelength_iterations.value_or(0U) << "\n";
  report_stream << "first_segment_chars=" << first_builder.get_segment_chars().size() << "\n";
  report_stream << "second_segment_chars=" << second_builder.get_segment_chars().size() << "\n";
  report_stream << "first_patterns=" << first_builder.get_buffering_patterns().size() << "\n";
  report_stream << "second_patterns=" << second_builder.get_buffering_patterns().size() << "\n";
  report_stream << "first_lattice=" << realtech_fixture::FormatSegmentCharLatticeSummary(first_summary, first_builder) << "\n";
  report_stream << "second_lattice=" << realtech_fixture::FormatSegmentCharLatticeSummary(second_summary, second_builder) << "\n";
  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("repeat_reduced_builds", "repeat_reduced_builds_report.txt", report_stream.str()));
}

}  // namespace
}  // namespace icts_test
