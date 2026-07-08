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
 * @file ScopedLogFileTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Regression coverage for scoped report log redirection in tests.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "Flow.hh"
#include "common/CTSTestRuntime.hh"
#include "common/io/TestArtifactIO.hh"
#include "common/logging/LogText.hh"
#include "common/logging/ScopedLogFile.hh"
#include "utils/logger/Schema.hh"

namespace icts_test {
namespace {

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

TEST(ScopedLogFileTest, NestedScopedLogFilesRestoreOuterDestination)
{
  const auto output_dir = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "common" / "logging" / "scoped_log_file");
  ASSERT_FALSE(output_dir.empty());

  const auto outer_log = output_dir / "outer_cts.log";
  const auto inner_log = output_dir / "inner_cts.log";
  {
    const common::logging::ScopedLogFile outer_guard(outer_log, "Outer CTS Report");
    auto& reporter = icts_test::runtime::CurrentRuntime().reporter;
    icts::EmitKeyValueTable(reporter, "Outer Before",
                            {
                                {"phase", "before"},
                            });
    {
      const common::logging::ScopedLogFile inner_guard(inner_log, "Inner CTS Report");
      icts::EmitKeyValueTable(reporter, "Inner Body",
                              {
                                  {"phase", "inside"},
                              });
    }
    icts::EmitKeyValueTable(reporter, "Outer After",
                            {
                                {"phase", "after"},
                            });
  }

  const auto outer_content = ReadTextFile(outer_log);
  const auto inner_content = ReadTextFile(inner_log);
  ASSERT_FALSE(outer_content.empty());
  ASSERT_FALSE(inner_content.empty());
  EXPECT_NE(outer_content.find("Outer Before"), std::string::npos);
  EXPECT_NE(outer_content.find("Outer After"), std::string::npos);
  EXPECT_EQ(outer_content.find("Inner Body"), std::string::npos);
  EXPECT_NE(inner_content.find("Inner Body"), std::string::npos);
  EXPECT_EQ(inner_content.find("Outer After"), std::string::npos);
}

TEST(ScopedLogFileTest, StageScopeSummaryOmitsElapsedTimeFromSchemaFile)
{
  const auto output_dir = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "common" / "logging" / "scoped_stage");
  ASSERT_FALSE(output_dir.empty());

  const auto log_path = output_dir / "cts.log";
  {
    const common::logging::ScopedLogFile log_guard(log_path, "Scoped Stage CTS Report");
    auto stage = icts_test::runtime::CurrentRuntime().reporter.beginStage("UnitStage", "exercise report finish");
    stage.finished();
  }

  const auto log_content = ReadTextFile(log_path);
  ASSERT_FALSE(log_content.empty());
  const auto stage_summary = common::logging::ExtractTextBlock(log_content, "UnitStage exercise report finish Summary");
  ASSERT_FALSE(stage_summary.empty());
  EXPECT_NE(stage_summary.find("status"), std::string::npos);
  EXPECT_NE(stage_summary.find("finished"), std::string::npos);
  EXPECT_EQ(stage_summary.find("outcome"), std::string::npos);
  EXPECT_EQ(stage_summary.find("elapsed_time"), std::string::npos);
}

TEST(ScopedLogFileTest, DetailReportReceivesDetailOnlyTablesAndSharedDiagnostics)
{
  const auto output_dir = common::io::PrepareCleanOutputDir(common::io::ResolveOutputDir() / "common" / "logging" / "detail_report");
  ASSERT_FALSE(output_dir.empty());

  const auto log_path = output_dir / "cts.log";
  const auto detail_log_path = output_dir / "cts_detail.log";
  {
    const common::logging::ScopedLogFile log_guard(log_path, "Detail Routing CTS Report");
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTableTo("Default Only", {{"scope", "default"}}, icts::ReportSink::kDefault);
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTableTo("Detail Only", {{"scope", "detail"}}, icts::ReportSink::kDetail);
    icts::EmitDiagnostic(icts_test::runtime::CurrentRuntime().reporter, icts::DiagnosticLevel::kWarning, "DetailRouting",
                         "shared diagnostic", {{"case", "detail_report"}});

    auto detail_stage = icts_test::runtime::CurrentRuntime().reporter.beginStage(
        "DetailStage", "trace", {{"phase", "detail"}},
        icts::StageReportOptions{.context_sink = icts::ReportSink::kDetail, .summary_sink = icts::ReportSink::kDetail});
    detail_stage.finished({{"count", "1"}});

    auto failed_detail_stage = icts_test::runtime::CurrentRuntime().reporter.beginStage(
        "DetailStage", "failure trace", {{"phase", "detail"}},
        icts::StageReportOptions{.context_sink = icts::ReportSink::kDetail, .summary_sink = icts::ReportSink::kDetail});
    failed_detail_stage.failed({{"reason", "unit_failure"}});
  }

  const auto log_content = ReadTextFile(log_path);
  const auto detail_log_content = ReadTextFile(detail_log_path);
  ASSERT_FALSE(log_content.empty());
  ASSERT_FALSE(detail_log_content.empty());
  EXPECT_NE(log_content.find("Default Only"), std::string::npos);
  EXPECT_EQ(log_content.find("Detail Only"), std::string::npos);
  EXPECT_EQ(log_content.find("DetailStage trace"), std::string::npos);
  EXPECT_EQ(log_content.find("DetailStage failure trace Context"), std::string::npos);
  EXPECT_NE(log_content.find("DetailStage failure trace Summary"), std::string::npos);
  EXPECT_NE(log_content.find("unit_failure"), std::string::npos);
  EXPECT_NE(log_content.find("DetailRouting Diagnostic"), std::string::npos);
  EXPECT_NE(detail_log_content.find("Detail Only"), std::string::npos);
  EXPECT_NE(detail_log_content.find("DetailStage trace Context"), std::string::npos);
  EXPECT_NE(detail_log_content.find("DetailStage trace Summary"), std::string::npos);
  EXPECT_NE(detail_log_content.find("DetailStage failure trace Context"), std::string::npos);
  EXPECT_NE(detail_log_content.find("DetailStage failure trace Summary"), std::string::npos);
  EXPECT_NE(detail_log_content.find("DetailRouting Diagnostic"), std::string::npos);
}

}  // namespace
}  // namespace icts_test
