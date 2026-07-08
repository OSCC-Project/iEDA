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
 * @file main.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @brief iCTS test entry point
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "Flow.hh"
#include "common/CTSTestRuntime.hh"
#include "common/io/TestArtifactIO.hh"
#include "utils/logger/Schema.hh"

namespace icts_test {
namespace {

class PerTestArtifactListener : public ::testing::EmptyTestEventListener
{
 public:
  explicit PerTestArtifactListener(std::string executable_name) : _executable_name(std::move(executable_name)) {}

  void OnTestStart(const ::testing::TestInfo& test_info) override
  {
    runtime::BeginCurrentRuntime();
    _current_output_dir = common::io::PrepareCleanOutputDir(
        common::io::ResolveOutputDir() / "gtest" / common::io::SanitizeOutputName(_executable_name)
        / common::io::SanitizeOutputName(test_info.test_suite_name()) / common::io::SanitizeOutputName(test_info.name()));
    _current_cts_log = _current_output_dir / "cts.log";
    _current_test_log = _current_output_dir / "test.log";

    icts_test::runtime::CurrentRuntime().reporter.open(_current_cts_log, "iCTS Test Report",
                                                       {
                                                           {"cts_log", _current_cts_log.string()},
                                                       });
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable("GTest Case Context", {
                                                                                              {"executable", _executable_name},
                                                                                              {"test_suite", test_info.test_suite_name()},
                                                                                              {"test_name", test_info.name()},
                                                                                              {"output_dir", _current_output_dir.string()},
                                                                                              {"test_log", _current_test_log.string()},
                                                                                          });

    ::testing::internal::CaptureStdout();
    ::testing::internal::CaptureStderr();
  }

  void OnTestEnd(const ::testing::TestInfo& test_info) override
  {
    const std::string captured_stdout = ::testing::internal::GetCapturedStdout();
    const std::string captured_stderr = ::testing::internal::GetCapturedStderr();

    const auto* result = test_info.result();
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable("GTest Case Result",
                                                                    {
                                                                        {"status", resolveTestStatus(*result)},
                                                                        {"elapsed_ms", std::to_string(result->elapsed_time())},
                                                                        {"total_part_count", std::to_string(result->total_part_count())},
                                                                        {"test_log", _current_test_log.string()},
                                                                    });
    icts_test::runtime::CurrentRuntime().reporter.close();
    runtime::EndCurrentRuntime();

    common::io::WriteRawTextLog(_current_test_log, buildTestLogContent(test_info, captured_stdout, captured_stderr));
    _current_output_dir.clear();
    _current_cts_log.clear();
    _current_test_log.clear();
  }

 private:
  static auto resolveTestStatus(const ::testing::TestResult& result) -> std::string
  {
    if (result.Passed()) {
      return "passed";
    }
    if (result.Skipped()) {
      return "skipped";
    }
    return "failed";
  }

  static auto buildTestLogContent(const ::testing::TestInfo& test_info, const std::string& captured_stdout,
                                  const std::string& captured_stderr) -> std::string
  {
    std::ostringstream stream;
    stream << "test_suite: " << test_info.test_suite_name() << "\n";
    stream << "test_name: " << test_info.name() << "\n";
    stream << "stdout_bytes: " << captured_stdout.size() << "\n";
    stream << "stderr_bytes: " << captured_stderr.size() << "\n";
    stream << "\n[stdout]\n";
    stream << (captured_stdout.empty() ? "(empty)\n" : captured_stdout);
    if (!captured_stdout.empty() && captured_stdout.back() != '\n') {
      stream << "\n";
    }
    stream << "\n[stderr]\n";
    stream << (captured_stderr.empty() ? "(empty)\n" : captured_stderr);
    if (!captured_stderr.empty() && captured_stderr.back() != '\n') {
      stream << "\n";
    }
    return stream.str();
  }

  std::string _executable_name;
  std::filesystem::path _current_output_dir;
  std::filesystem::path _current_cts_log;
  std::filesystem::path _current_test_log;
};

auto ResolveExecutableName(const char* argv0) -> std::string
{
  if (argv0 == nullptr || *argv0 == '\0') {
    return "icts_test";
  }
  const auto executable_path = std::filesystem::path(argv0);
  const auto filename = executable_path.filename().string();
  return filename.empty() ? "icts_test" : filename;
}

}  // namespace
}  // namespace icts_test

auto main(int argc, char** argv) -> int
{
  ::testing::InitGoogleTest(&argc, argv);
  auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
  const char* executable_path = nullptr;
  if (argc > 0 && argv != nullptr) {
    executable_path = *argv;
  }
  auto per_test_artifact_listener = std::make_unique<icts_test::PerTestArtifactListener>(icts_test::ResolveExecutableName(executable_path));
  listeners.Append(per_test_artifact_listener.release());
  return RUN_ALL_TESTS();
}
