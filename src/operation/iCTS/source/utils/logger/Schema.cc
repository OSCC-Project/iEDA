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
 * @file Schema.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Structured report writer for iCTS runtime reports and generated artifact references.
 */

#include "Schema.hh"

#include <glog/logging.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "Log.hh"

namespace icts {
namespace {

auto BuildGeneratedOnString() -> std::string
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t time_value = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &time_value);
#else
  localtime_r(&time_value, &local_tm);
#endif

  std::ostringstream stream;
  stream << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
  return stream.str();
}

auto BuildRunHeader(const std::string& run_title) -> std::string
{
  std::ostringstream stream;
  stream << "# " << (run_title.empty() ? "iCTS Report" : run_title) << '\n';
  stream << "Generate the report at " << BuildGeneratedOnString() << '\n';
  return stream.str();
}

auto BuildDetailPath(const std::filesystem::path& path) -> std::filesystem::path
{
  if (path.empty()) {
    return {};
  }
  const auto parent = path.parent_path();
  const auto stem = path.stem().string();
  const auto extension = path.extension().string();
  const auto detail_filename = stem.empty() ? std::string{"cts_detail"} + extension : stem + "_detail" + extension;
  return parent.empty() ? std::filesystem::path(detail_filename) : parent / detail_filename;
}

auto FilterRunMetadata(const KeyValueFields& metadata) -> KeyValueFields
{
  KeyValueFields filtered_metadata;
  filtered_metadata.reserve(metadata.size());
  for (const auto& [key, value] : metadata) {
    if (key == "generated_on" || key == "work_dir") {
      continue;
    }
    filtered_metadata.emplace_back(key, value);
  }
  return filtered_metadata;
}

auto BuildDetailBlock(const std::string& title, const std::vector<std::string>& lines) -> std::string
{
  std::ostringstream stream;
  stream << logformat::MakeTitle(title);
  if (lines.empty()) {
    stream << "\n(empty)";
    return stream.str();
  }
  stream << '\n';
  for (const auto& line : lines) {
    stream << line << '\n';
  }
  std::string block = stream.str();
  if (!block.empty() && block.back() == '\n') {
    block.pop_back();
  }
  return block;
}

auto BuildSectionBlock(const std::string& title) -> std::string
{
  if (!title.empty() && title.front() == '#') {
    return title;
  }
  return logformat::MakeTitle(title);
}

auto BuildDiagnosticFields(DiagnosticLevel level, const std::string& owner, const std::string& summary, const KeyValueFields& fields)
    -> KeyValueFields
{
  KeyValueFields diagnostic_fields;
  diagnostic_fields.reserve(fields.size() + 3U);

  std::string severity = "info";
  switch (level) {
    case DiagnosticLevel::kInfo:
      severity = "info";
      break;
    case DiagnosticLevel::kWarning:
      severity = "warning";
      break;
    case DiagnosticLevel::kError:
      severity = "error";
      break;
    case DiagnosticLevel::kDegraded:
      severity = "degraded";
      break;
  }

  diagnostic_fields.emplace_back("severity", severity);
  diagnostic_fields.emplace_back("owner", owner);
  diagnostic_fields.emplace_back("summary", summary);
  diagnostic_fields.insert(diagnostic_fields.end(), fields.begin(), fields.end());
  return diagnostic_fields;
}

auto EnsureParentDir(const std::filesystem::path& path) -> bool
{
  std::error_code error_code;
  const auto parent = path.parent_path();
  if (parent.empty()) {
    return true;
  }
  std::filesystem::create_directories(parent, error_code);
  return !error_code;
}

auto AppendBlockToPath(const std::filesystem::path& path, const std::string& run_title, const std::string& block) -> void
{
  if (path.empty() || block.empty() || !EnsureParentDir(path)) {
    return;
  }

  std::error_code error_code;
  const bool file_exists = std::filesystem::exists(path, error_code) && !error_code;
  const auto size = file_exists ? std::filesystem::file_size(path, error_code) : 0U;
  const bool needs_header = !file_exists || error_code || size == 0U;

  std::ofstream stream(path, std::ios::out | std::ios::app);
  if (!stream.is_open()) {
    LOG_WARNING << "SchemaWriter: cannot open report file " << path.string();
    return;
  }

  if (needs_header) {
    stream << BuildRunHeader(run_title);
  }
  stream << '\n' << block;
  if (block.back() != '\n') {
    stream << '\n';
  }
  stream.flush();
}

}  // namespace

auto SchemaWriter::open(const std::filesystem::path& path, const std::string& run_title, const KeyValueFields& metadata) -> void
{
  if (path.empty()) {
    return;
  }

  const std::scoped_lock lock(_mutex);
  if (!EnsureParentDir(path)) {
    LOG_WARNING << "SchemaWriter: cannot create report directory for " << path.string();
    return;
  }

  std::ofstream next_stream(path, std::ios::out | std::ios::trunc);
  if (!next_stream.is_open()) {
    LOG_WARNING << "SchemaWriter: cannot open report file " << path.string();
    return;
  }
  const auto next_detail_path = BuildDetailPath(path);
  std::ofstream next_detail_stream;
  if (!next_detail_path.empty()) {
    next_detail_stream.open(next_detail_path, std::ios::out | std::ios::trunc);
    if (!next_detail_stream.is_open()) {
      LOG_WARNING << "SchemaWriter: cannot open detail report file " << next_detail_path.string();
    }
  }

  next_stream << BuildRunHeader(run_title);
  if (next_detail_stream.is_open()) {
    next_detail_stream << BuildRunHeader(run_title + " Detail");
  }
  const KeyValueFields filtered_metadata = FilterRunMetadata(metadata);
  if (!filtered_metadata.empty()) {
    next_stream << '\n' << logformat::MakeKeyValueTable("Run Context", filtered_metadata) << '\n';
    if (next_detail_stream.is_open()) {
      next_detail_stream << '\n' << logformat::MakeKeyValueTable("Run Context", filtered_metadata) << '\n';
    }
  }
  next_stream.flush();
  if (next_detail_stream.is_open()) {
    next_detail_stream.flush();
  }

  if (_stream.is_open()) {
    _suspended_writers.push_back(SuspendedWriter{
        .path = _path,
        .has_content = _has_content,
        .detail_path = _detail_path,
        .detail_has_content = _detail_has_content,
    });
    _stream.flush();
    _stream.close();
    if (_detail_stream.is_open()) {
      _detail_stream.flush();
      _detail_stream.close();
    }
  }

  _stream = std::move(next_stream);
  _path = path;
  _has_content = true;
  _detail_stream = std::move(next_detail_stream);
  _detail_path = _detail_stream.is_open() ? next_detail_path : std::filesystem::path{};
  _detail_has_content = _detail_stream.is_open();
}

auto SchemaWriter::close() -> void
{
  const std::scoped_lock lock(_mutex);
  if (_stream.is_open()) {
    _stream.flush();
    _stream.close();
  }
  if (_detail_stream.is_open()) {
    _detail_stream.flush();
    _detail_stream.close();
  }
  restoreSuspendedWriterLocked();
}

auto SchemaWriter::reset() -> void
{
  const std::scoped_lock lock(_mutex);
  if (_stream.is_open()) {
    _stream.flush();
    _stream.close();
  }
  if (_detail_stream.is_open()) {
    _detail_stream.flush();
    _detail_stream.close();
  }
  _path.clear();
  _has_content = false;
  _detail_path.clear();
  _detail_has_content = false;
  _suspended_writers.clear();
  _runtime_metrics.clear();
}

auto SchemaWriter::restoreSuspendedWriterLocked() -> void
{
  if (_suspended_writers.empty()) {
    _path.clear();
    _has_content = false;
    _detail_path.clear();
    _detail_has_content = false;
    return;
  }

  const auto suspended_writer = _suspended_writers.back();
  _suspended_writers.pop_back();

  std::ofstream restored_stream(suspended_writer.path, std::ios::out | std::ios::app);
  if (!restored_stream.is_open()) {
    LOG_WARNING << "SchemaWriter: cannot restore suspended report file " << suspended_writer.path.string();
    _path.clear();
    _has_content = false;
    return;
  }

  _stream = std::move(restored_stream);
  _path = suspended_writer.path;
  _has_content = suspended_writer.has_content;

  if (!suspended_writer.detail_path.empty()) {
    std::ofstream restored_detail_stream(suspended_writer.detail_path, std::ios::out | std::ios::app);
    if (!restored_detail_stream.is_open()) {
      LOG_WARNING << "SchemaWriter: cannot restore suspended detail report file " << suspended_writer.detail_path.string();
      _detail_path.clear();
      _detail_has_content = false;
      return;
    }
    _detail_stream = std::move(restored_detail_stream);
    _detail_path = suspended_writer.detail_path;
    _detail_has_content = suspended_writer.detail_has_content;
  } else {
    _detail_path.clear();
    _detail_has_content = false;
  }
}

auto SchemaWriter::isOpen() const -> bool
{
  const std::scoped_lock lock(_mutex);
  return _stream.is_open();
}

auto SchemaWriter::getActivePath() const -> std::filesystem::path
{
  const std::scoped_lock lock(_mutex);
  return _path;
}

auto SchemaWriter::getDetailPath() const -> std::filesystem::path
{
  const std::scoped_lock lock(_mutex);
  return _detail_path;
}

auto SchemaWriter::writeBlockToStream(std::ofstream& stream, bool& has_content, const std::string& block) -> void
{
  if (!stream.is_open() || block.empty()) {
    return;
  }
  if (has_content) {
    stream << '\n';
  }
  stream << block;
  if (block.back() != '\n') {
    stream << '\n';
  }
  stream.flush();
  has_content = true;
}

auto SchemaWriter::writeBlockLocked(const std::string& block, ReportSink sink) -> void
{
  if (block.empty() || sink == ReportSink::kNone) {
    return;
  }
  if (sink == ReportSink::kDefault || sink == ReportSink::kBoth) {
    writeBlockToStream(_stream, _has_content, block);
  }
  if (sink == ReportSink::kDetail || sink == ReportSink::kBoth) {
    writeBlockToStream(_detail_stream, _detail_has_content, block);
  }
}

auto SchemaWriter::emitSection(const std::string& title) -> void
{
  emitSectionTo(title, ReportSink::kDefault);
}

auto SchemaWriter::emitSectionTo(const std::string& title, ReportSink sink) -> void
{
  const std::scoped_lock lock(_mutex);
  writeBlockLocked(BuildSectionBlock(title), sink);
}

auto SchemaWriter::emitTable(const std::string& title, const std::vector<std::string>& headers, const TableRows& rows) -> void
{
  emitTableTo(title, headers, rows, ReportSink::kDefault);
}

auto SchemaWriter::emitTableTo(const std::string& title, const std::vector<std::string>& headers, const TableRows& rows, ReportSink sink)
    -> void
{
  const std::scoped_lock lock(_mutex);
  writeBlockLocked(logformat::MakeTitledTable(title, headers, rows), sink);
}

auto SchemaWriter::emitKeyValueTable(const std::string& title, const KeyValueFields& fields) -> void
{
  emitKeyValueTableTo(title, fields, ReportSink::kDefault);
}

auto SchemaWriter::emitKeyValueTableTo(const std::string& title, const KeyValueFields& fields, ReportSink sink) -> void
{
  const std::scoped_lock lock(_mutex);
  writeBlockLocked(logformat::MakeKeyValueTable(title, fields), sink);
}

auto SchemaWriter::emitDetailBlock(const std::string& title, const std::vector<std::string>& lines) -> void
{
  emitDetailBlockTo(title, lines, ReportSink::kDefault);
}

auto SchemaWriter::emitDetailBlockTo(const std::string& title, const std::vector<std::string>& lines, ReportSink sink) -> void
{
  const std::scoped_lock lock(_mutex);
  writeBlockLocked(BuildDetailBlock(title, lines), sink);
}

auto SchemaWriter::emitDiagnostic(DiagnosticLevel level, const std::string& owner, const std::string& summary, const KeyValueFields& fields)
    -> void
{
  emitKeyValueTableTo(owner + " Diagnostic", BuildDiagnosticFields(level, owner, summary, fields), ReportSink::kBoth);
}

auto SchemaWriter::emitArtifact(const std::string& label, const std::filesystem::path& path, const std::string& detail) -> void
{
  emitArtifactTo(label, path, detail, ReportSink::kBoth);
}

auto SchemaWriter::emitArtifactTo(const std::string& label, const std::filesystem::path& path, const std::string& detail, ReportSink sink)
    -> void
{
  KeyValueFields fields = {
      {"label", label},
      {"path", path.string()},
  };
  if (!detail.empty()) {
    fields.emplace_back("detail", detail);
  }
  emitKeyValueTableTo("Generated Artifact", fields, sink);
}

auto SchemaWriter::appendStandaloneTable(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                         const std::vector<std::string>& headers, const TableRows& rows) -> void
{
  appendStandaloneBlock(path, run_title, logformat::MakeTitledTable(title, headers, rows));
}

auto SchemaWriter::appendStandaloneKeyValueTable(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                                 const KeyValueFields& fields) -> void
{
  appendStandaloneBlock(path, run_title, logformat::MakeKeyValueTable(title, fields));
}

auto SchemaWriter::appendStandaloneDetailBlock(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                               const std::vector<std::string>& lines) -> void
{
  appendStandaloneBlock(path, run_title, BuildDetailBlock(title, lines));
}

auto SchemaWriter::appendStandaloneArtifact(const std::filesystem::path& path, const std::string& run_title, const std::string& label,
                                            const std::filesystem::path& artifact_path, const std::string& detail) -> void
{
  KeyValueFields fields = {
      {"label", label},
      {"path", artifact_path.string()},
  };
  if (!detail.empty()) {
    fields.emplace_back("detail", detail);
  }
  appendStandaloneKeyValueTable(path, run_title, "Generated Artifact", fields);
}

auto SchemaWriter::appendStandaloneBlock(const std::filesystem::path& path, const std::string& run_title, const std::string& block) -> void
{
  static std::mutex append_mutex;
  const std::scoped_lock lock(append_mutex);
  AppendBlockToPath(path, run_title, block);
}

auto EmitTable(SchemaWriter& writer, const std::string& title, const std::vector<std::string>& headers, const TableRows& rows) -> void
{
  LOG_INFO << "";
  LOG_INFO << logformat::MakeTitledTable(title, headers, rows);
  writer.emitTable(title, headers, rows);
}

auto EmitKeyValueTable(SchemaWriter& writer, const std::string& title, const KeyValueFields& fields) -> void
{
  LOG_INFO << "";
  LOG_INFO << logformat::MakeKeyValueTable(title, fields);
  writer.emitKeyValueTable(title, fields);
}

auto EmitDiagnostic(SchemaWriter& writer, DiagnosticLevel level, const std::string& owner, const std::string& summary,
                    const KeyValueFields& fields) -> void
{
  switch (level) {
    case DiagnosticLevel::kInfo:
      LOG_INFO << owner << ": " << summary;
      break;
    case DiagnosticLevel::kWarning:
    case DiagnosticLevel::kDegraded:
      LOG_WARNING << owner << ": " << summary;
      break;
    case DiagnosticLevel::kError:
      LOG_ERROR << owner << ": " << summary;
      break;
  }
  writer.emitDiagnostic(level, owner, summary, fields);
}

auto EmitArtifact(SchemaWriter& writer, const std::string& label, const std::filesystem::path& path, const std::string& detail) -> void
{
  LOG_INFO << label << " saved: " << path.string();
  writer.emitArtifact(label, path, detail);
}

}  // namespace icts
