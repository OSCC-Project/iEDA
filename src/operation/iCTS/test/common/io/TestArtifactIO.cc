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
 * @file TestArtifactIO.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared artifact IO and per-test report helpers for iCTS tests.
 */

#include "common/io/TestArtifactIO.hh"

#include <glog/logging.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "Flow.hh"
#include "Log.hh"
#include "common/CTSTestRuntime.hh"
#include "common/dataset/TestDataset.hh"
#include "utils/logger/LogFormat.hh"
#include "utils/logger/Schema.hh"

namespace icts_test::common::io {
namespace {

struct ParsedInfoReport
{
  std::string title;
  std::vector<std::pair<std::string, std::string>> key_value_lines;
  std::vector<std::string> detail_lines;
};

constexpr std::size_t kConsoleTitleDisplayLimit = 96U;
constexpr std::size_t kConsoleKeyDisplayLimit = 48U;
constexpr std::size_t kConsoleValueDisplayLimit = 160U;
constexpr std::size_t kConsoleDetailDisplayLimit = 200U;
constexpr std::string_view kConsoleTruncationMarker = "...";

auto ResolveExecutableDir() -> std::filesystem::path
{
  std::error_code error_code;
  const auto executable_path = std::filesystem::canonical("/proc/self/exe", error_code);
  if (error_code || executable_path.empty()) {
    return {};
  }
  return executable_path.parent_path();
}

auto TrimWhitespace(const std::string& line) -> std::string
{
  const auto begin = line.find_first_not_of(" \t\r");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = line.find_last_not_of(" \t\r");
  return line.substr(begin, end - begin + 1U);
}

auto TryParseKeyValueLine(const std::string& line, std::string& key, std::string& value) -> bool
{
  const auto colon_position = line.find(": ");
  if (colon_position != std::string::npos && colon_position > 0U) {
    key = TrimWhitespace(line.substr(0, colon_position));
    value = TrimWhitespace(line.substr(colon_position + 2U));
    return !key.empty();
  }

  const auto equal_position = line.find('=');
  if (equal_position == std::string::npos || equal_position == 0U) {
    return false;
  }
  if (line.find('=', equal_position + 1U) != std::string::npos) {
    return false;
  }

  key = TrimWhitespace(line.substr(0, equal_position));
  value = TrimWhitespace(line.substr(equal_position + 1U));
  if (key.empty()) {
    return false;
  }
  if (key.front() == '-' || key.front() == '*') {
    return false;
  }
  return true;
}

auto ParseInfoReport(const InfoReport& report) -> ParsedInfoReport
{
  ParsedInfoReport parsed_report;
  parsed_report.title = report.title.empty() ? "artifact_report" : report.title;

  std::istringstream input_stream(report.content);
  for (std::string line; std::getline(input_stream, line);) {
    const auto trimmed = TrimWhitespace(line);
    if (trimmed.empty()) {
      continue;
    }

    std::string key;
    std::string value;
    if (TryParseKeyValueLine(trimmed, key, value)) {
      parsed_report.key_value_lines.emplace_back(std::move(key), std::move(value));
      continue;
    }
    parsed_report.detail_lines.push_back(trimmed);
  }
  return parsed_report;
}

auto BuildInfoReportBlock(const ParsedInfoReport& report) -> std::string
{
  std::ostringstream output_stream;
  output_stream << '\n';

  if (!report.key_value_lines.empty()) {
    output_stream << icts::logformat::MakeKeyValueTable(report.title, report.key_value_lines);
  } else {
    output_stream << icts::logformat::MakeTitle(report.title);
  }

  if (!report.detail_lines.empty()) {
    output_stream << "\n-- details --\n";
    for (const auto& detail : report.detail_lines) {
      output_stream << detail << "\n";
    }
  }

  if (report.key_value_lines.empty() && report.detail_lines.empty()) {
    output_stream << "\n(empty report)\n";
  }
  return output_stream.str();
}

auto TruncateConsoleDisplayText(const std::string& text, std::size_t max_length) -> std::string
{
  if (max_length == 0U || text.size() <= max_length) {
    return text;
  }
  if (max_length <= kConsoleTruncationMarker.size()) {
    return std::string(kConsoleTruncationMarker.substr(0U, max_length));
  }

  const auto remaining_length = max_length - kConsoleTruncationMarker.size();
  const auto head_length = remaining_length / 2U;
  const auto tail_length = remaining_length - head_length;
  return text.substr(0U, head_length) + std::string(kConsoleTruncationMarker) + text.substr(text.size() - tail_length);
}

auto BuildConsoleDisplayReport(const ParsedInfoReport& report) -> ParsedInfoReport
{
  ParsedInfoReport console_report;
  console_report.title = TruncateConsoleDisplayText(report.title, kConsoleTitleDisplayLimit);
  console_report.key_value_lines.reserve(report.key_value_lines.size());
  for (const auto& [key, value] : report.key_value_lines) {
    console_report.key_value_lines.emplace_back(TruncateConsoleDisplayText(key, kConsoleKeyDisplayLimit),
                                                TruncateConsoleDisplayText(value, kConsoleValueDisplayLimit));
  }

  console_report.detail_lines.reserve(report.detail_lines.size());
  for (const auto& detail : report.detail_lines) {
    console_report.detail_lines.push_back(TruncateConsoleDisplayText(detail, kConsoleDetailDisplayLimit));
  }
  return console_report;
}

auto EmitParsedInfoReport(const ParsedInfoReport& report) -> void
{
  if (!report.key_value_lines.empty()) {
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable(report.title, report.key_value_lines);
  } else {
    icts_test::runtime::CurrentRuntime().reporter.emitSection(report.title);
  }

  if (!report.detail_lines.empty()) {
    icts_test::runtime::CurrentRuntime().reporter.emitDetailBlock(report.title + " Details", report.detail_lines);
  }
}

auto IsActiveReportPath(const std::filesystem::path& path) -> bool
{
  auto& schema_writer = icts_test::runtime::CurrentRuntime().reporter;
  return schema_writer.isOpen() && schema_writer.getActivePath() == path;
}

auto EmitOrAppendTestKeyValueTable(const std::filesystem::path& path, const std::string& title, const icts::KeyValueFields& fields) -> void
{
  if (IsActiveReportPath(path)) {
    icts_test::runtime::CurrentRuntime().reporter.emitKeyValueTable(title, fields);
    return;
  }
  icts::SchemaWriter::appendStandaloneKeyValueTable(path, kDefaultTestReportTitle, title, fields);
}

auto EmitOrAppendTestDetailBlock(const std::filesystem::path& path, const std::string& title, const std::vector<std::string>& lines) -> void
{
  if (IsActiveReportPath(path)) {
    icts_test::runtime::CurrentRuntime().reporter.emitDetailBlock(title, lines);
    return;
  }
  icts::SchemaWriter::appendStandaloneDetailBlock(path, kDefaultTestReportTitle, title, lines);
}

auto MirrorStandaloneTextLog(const std::filesystem::path& path, const std::string& content) -> void
{
  if (path.filename() == "cts.log") {
    return;
  }

  const InfoReport report{.title = path.filename().string(), .content = content};
  const auto parsed_report = ParseInfoReport(report);
  const auto cts_log_path = path.parent_path() / "cts.log";
  if (!parsed_report.key_value_lines.empty()) {
    EmitOrAppendTestKeyValueTable(cts_log_path, parsed_report.title, parsed_report.key_value_lines);
  }
  if (!parsed_report.detail_lines.empty()) {
    const std::vector<std::string> detail_summary = {
        "detail_lines=" + std::to_string(parsed_report.detail_lines.size()),
        "details omitted from cts.log; see artifact file: " + path.string(),
    };
    EmitOrAppendTestDetailBlock(cts_log_path, parsed_report.title + " Details", detail_summary);
  }
  if (parsed_report.key_value_lines.empty() && parsed_report.detail_lines.empty()) {
    EmitOrAppendTestDetailBlock(cts_log_path, parsed_report.title, {"(empty report)"});
  }
}

}  // namespace

auto WriteRawTextLog(const std::filesystem::path& path, const std::string& content) -> bool
{
  std::error_code error_code;
  const auto parent_dir = path.parent_path();
  if (!parent_dir.empty()) {
    std::filesystem::create_directories(parent_dir, error_code);
    if (error_code) {
      return false;
    }
  }

  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return false;
  }
  output_stream << content;
  return true;
}

auto WriteTextLog(const std::filesystem::path& path, const std::string& content) -> bool
{
  if (!WriteRawTextLog(path, content)) {
    return false;
  }
  MirrorStandaloneTextLog(path, content);
  return true;
}

auto EmitInfoReport(const InfoReport& report) -> void
{
  const auto parsed_report = ParseInfoReport(report);
  LOG_INFO << BuildInfoReportBlock(BuildConsoleDisplayReport(parsed_report));
  EmitParsedInfoReport(parsed_report);
}

auto OpenTestReport(const std::filesystem::path& path, const std::string& run_title) -> void
{
  icts_test::runtime::CurrentRuntime().reporter.open(path, run_title,
                                                     {
                                                         {"cts_log", path.string()},
                                                     });
}

auto CloseTestReport() -> void
{
  icts_test::runtime::CurrentRuntime().reporter.close();
}

auto SanitizeOutputName(const std::string& raw_name) -> std::string
{
  std::string sanitized;
  sanitized.reserve(raw_name.size());

  bool previous_was_separator = false;
  for (const char value : raw_name) {
    const auto character = static_cast<unsigned char>(value);
    if (std::isalnum(character) != 0) {
      sanitized.push_back(static_cast<char>(std::tolower(character)));
      previous_was_separator = false;
      continue;
    }

    if (!previous_was_separator && !sanitized.empty()) {
      sanitized.push_back('_');
      previous_was_separator = true;
    }
  }

  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }
  return sanitized.empty() ? "unnamed" : sanitized;
}

auto PrepareCleanOutputDir(const std::filesystem::path& path) -> std::filesystem::path
{
  std::error_code error_code;
  std::filesystem::remove_all(path, error_code);
  if (error_code) {
    return {};
  }

  error_code.clear();
  std::filesystem::create_directories(path, error_code);
  if (error_code) {
    return {};
  }
  return path;
}

auto ResolveOutputDir() -> std::filesystem::path
{
  const char* env_dir = std::getenv("ICTS_TEST_OUTPUT_DIR");
  if (env_dir != nullptr && *env_dir != '\0') {
    return {env_dir};
  }

  // Default to an executable-adjacent output root so test artifacts stay stable
  // regardless of the caller's current working directory.
  const auto executable_dir = ResolveExecutableDir();
  if (!executable_dir.empty()) {
    return executable_dir / "icts_test_output";
  }

  return {"icts_test_output"};
}

auto ResolveTopologyGenOutputDir() -> std::filesystem::path
{
  return ResolveOutputDir() / "topology_gen";
}

auto ResolveClusteringOutputDir() -> std::filesystem::path
{
  return ResolveOutputDir() / "clustering";
}

}  // namespace icts_test::common::io
