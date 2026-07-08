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
 * @file TestArtifactIOTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Regression coverage for console-facing report truncation.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "common/dataset/TestDataset.hh"
#include "common/io/TestArtifactIO.hh"

namespace icts_test {
namespace {

constexpr const char* kExecutableName = "icts_test_common_io_artifact";
constexpr const char* kSuiteName = "TestArtifactIOTest";
constexpr const char* kEmitCaseName = "A_LongConsoleTablesKeepShapeAndTruncateValues";
constexpr const char* kVisiblePrefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
constexpr const char* kVisibleSuffix = "tail_marker_xyz";

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

auto BuildEmitCaseOutputDir() -> std::filesystem::path
{
  return common::io::ResolveOutputDir() / "gtest" / common::io::SanitizeOutputName(kExecutableName)
         / common::io::SanitizeOutputName(kSuiteName) / common::io::SanitizeOutputName(kEmitCaseName);
}

TEST(TestArtifactIOTest, A_LongConsoleTablesKeepShapeAndTruncateValues)
{
  std::string long_value = kVisiblePrefix;
  long_value.append(256U, 'x');
  long_value += kVisibleSuffix;

  common::io::EmitInfoReport(InfoReport{
      .title = "console_truncation_probe",
      .content = "long_field: " + long_value
                 + "\n"
                   "detail line without separator should also stay readable: "
                 + long_value + "\n",
  });

  const auto cts_log_path = BuildEmitCaseOutputDir() / "cts.log";
  const auto cts_log_content = ReadTextFile(cts_log_path);
  EXPECT_NE(cts_log_content.find(long_value), std::string::npos);
}

TEST(TestArtifactIOTest, B_PreviousCaseTestLogUsesEllipsisForLongValues)
{
  const auto test_log_path = BuildEmitCaseOutputDir() / "test.log";
  const auto test_log_content = ReadTextFile(test_log_path);
  ASSERT_FALSE(test_log_content.empty()) << "Missing test.log: " << test_log_path.string();

  EXPECT_NE(test_log_content.find("console_truncation_probe"), std::string::npos);
  EXPECT_NE(test_log_content.find("| Field"), std::string::npos);
  EXPECT_NE(test_log_content.find("| long_field"), std::string::npos);
  EXPECT_NE(test_log_content.find("long_field"), std::string::npos);
  EXPECT_NE(test_log_content.find(std::string(kVisiblePrefix)), std::string::npos);
  EXPECT_NE(test_log_content.find("..."), std::string::npos);
  EXPECT_NE(test_log_content.find(kVisibleSuffix), std::string::npos);
  EXPECT_NE(test_log_content.find("detail line without se...ould also stay readable"), std::string::npos);
  EXPECT_EQ(test_log_content.find(std::string(256U, 'x')), std::string::npos);
}

}  // namespace
}  // namespace icts_test
