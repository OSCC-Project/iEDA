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
 * @file Schema.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Structured report writer for iCTS runtime reports and generated artifact references.
 */

#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "LogFormat.hh"

namespace ieda {
class Stats;
}  // namespace ieda

namespace icts {

using KeyValueFields = std::vector<std::pair<std::string, std::string>>;
using TableRows = logformat::TableRows;

enum class DiagnosticLevel
{
  kInfo,
  kWarning,
  kError,
  kDegraded
};

enum class ReportSink
{
  kDefault,
  kDetail,
  kBoth,
  kNone
};

struct StageReportOptions
{
  ReportSink context_sink = ReportSink::kDefault;
  ReportSink summary_sink = ReportSink::kDefault;
  bool emit_success_summary = true;
};

class SchemaWriter
{
 public:
  struct RuntimeMetricRecord
  {
    double elapsed_time_s = 0.0;
    double peak_vmem_delta_mb = 0.0;
  };

  SchemaWriter() = default;
  ~SchemaWriter() = default;

  class RuntimeMetricScope
  {
   public:
    RuntimeMetricScope(const RuntimeMetricScope&) = delete;
    RuntimeMetricScope(RuntimeMetricScope&& other) noexcept;
    auto operator=(const RuntimeMetricScope&) -> RuntimeMetricScope& = delete;
    auto operator=(RuntimeMetricScope&& other) noexcept -> RuntimeMetricScope&;
    ~RuntimeMetricScope();

    auto finish(const std::string& status) -> RuntimeMetricRecord;
    auto finished() -> RuntimeMetricRecord;
    auto failed() -> RuntimeMetricRecord;
    auto measure() const -> RuntimeMetricRecord;

   private:
    friend class SchemaWriter;

    RuntimeMetricScope(SchemaWriter& writer, std::string stage);

    SchemaWriter* _writer = nullptr;
    std::string _stage;
    std::unique_ptr<ieda::Stats> _stats;
    bool _finished = false;
  };

  class StageScope
  {
   public:
    StageScope(const StageScope&) = delete;
    StageScope(StageScope&& other) noexcept;
    auto operator=(const StageScope&) -> StageScope& = delete;
    auto operator=(StageScope&& other) noexcept -> StageScope&;
    ~StageScope();

    auto markRunning(const std::string& summary, const KeyValueFields& fields = {}) -> void;
    auto finished(const KeyValueFields& finish_fields = {}) -> void;
    auto failed(const KeyValueFields& finish_fields = {}) -> void;
    auto skip(const KeyValueFields& finish_fields = {}) -> void;

   private:
    friend class SchemaWriter;

    StageScope(SchemaWriter& writer, std::string module, std::string stage, const KeyValueFields& start_fields = {},
               StageReportOptions report_options = {});

    auto closeWithStatus(const std::string& status, const KeyValueFields& finish_fields) -> void;

    std::string _module;
    std::string _stage;
    StageReportOptions _report_options;
    SchemaWriter* _writer = nullptr;
    std::chrono::steady_clock::time_point _start_time;
    bool _finished = false;
  };

  auto open(const std::filesystem::path& path, const std::string& run_title, const KeyValueFields& metadata = {}) -> void;
  // Close the active output and restore any suspended nested writer.
  auto close() -> void;
  // API teardown: close output, drop nested writer state, and clear run metrics.
  auto reset() -> void;
  auto isOpen() const -> bool;
  auto getActivePath() const -> std::filesystem::path;
  auto getDetailPath() const -> std::filesystem::path;

  auto emitSection(const std::string& title) -> void;
  auto emitSectionTo(const std::string& title, ReportSink sink) -> void;
  auto emitTable(const std::string& title, const std::vector<std::string>& headers, const TableRows& rows) -> void;
  auto emitTableTo(const std::string& title, const std::vector<std::string>& headers, const TableRows& rows, ReportSink sink) -> void;
  auto emitKeyValueTable(const std::string& title, const KeyValueFields& fields) -> void;
  auto emitKeyValueTableTo(const std::string& title, const KeyValueFields& fields, ReportSink sink) -> void;
  auto emitDetailBlock(const std::string& title, const std::vector<std::string>& lines) -> void;
  auto emitDetailBlockTo(const std::string& title, const std::vector<std::string>& lines, ReportSink sink) -> void;
  auto emitDiagnostic(DiagnosticLevel level, const std::string& owner, const std::string& summary, const KeyValueFields& fields = {})
      -> void;
  auto emitArtifact(const std::string& label, const std::filesystem::path& path, const std::string& detail = {}) -> void;
  auto emitArtifactTo(const std::string& label, const std::filesystem::path& path, const std::string& detail, ReportSink sink) -> void;
  auto resetRuntimeMetrics() -> void;
  auto beginRuntimeMetric(std::string stage) -> RuntimeMetricScope;
  auto emitRuntimeSummary(const std::string& title = "Runtime Summary") -> void;
  auto emitRuntimeMetricTable(const std::string& title, const std::string& stage, const std::string& status,
                              const RuntimeMetricRecord& metric_record) -> void;
  auto beginStage(std::string module, std::string stage, const KeyValueFields& start_fields = {}) -> StageScope;
  auto beginStage(std::string module, std::string stage, const KeyValueFields& start_fields, StageReportOptions report_options)
      -> StageScope;

  static auto appendStandaloneTable(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                    const std::vector<std::string>& headers, const TableRows& rows) -> void;
  static auto appendStandaloneKeyValueTable(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                            const KeyValueFields& fields) -> void;
  static auto appendStandaloneDetailBlock(const std::filesystem::path& path, const std::string& run_title, const std::string& title,
                                          const std::vector<std::string>& lines) -> void;
  static auto appendStandaloneArtifact(const std::filesystem::path& path, const std::string& run_title, const std::string& label,
                                       const std::filesystem::path& artifact_path, const std::string& detail = {}) -> void;

 private:
  struct SuspendedWriter
  {
    std::filesystem::path path;
    bool has_content = false;
    std::filesystem::path detail_path;
    bool detail_has_content = false;
  };

  struct RuntimeMetric
  {
    std::string stage;
    std::string status;
    double elapsed_time_s = 0.0;
    double peak_vmem_delta_mb = 0.0;
  };

  auto writeBlockLocked(const std::string& block, ReportSink sink = ReportSink::kDefault) -> void;
  static auto writeBlockToStream(std::ofstream& stream, bool& has_content, const std::string& block) -> void;
  auto recordRuntimeMetric(std::string stage, std::string status, const RuntimeMetricRecord& metric_record) -> void;
  auto restoreSuspendedWriterLocked() -> void;
  static auto appendStandaloneBlock(const std::filesystem::path& path, const std::string& run_title, const std::string& block) -> void;

  mutable std::mutex _mutex;
  std::ofstream _stream;
  std::filesystem::path _path;
  bool _has_content = false;
  std::ofstream _detail_stream;
  std::filesystem::path _detail_path;
  bool _detail_has_content = false;
  std::vector<SuspendedWriter> _suspended_writers;
  std::vector<RuntimeMetric> _runtime_metrics;
};

auto EmitTable(SchemaWriter& writer, const std::string& title, const std::vector<std::string>& headers, const TableRows& rows) -> void;
auto EmitKeyValueTable(SchemaWriter& writer, const std::string& title, const KeyValueFields& fields) -> void;
auto EmitDiagnostic(SchemaWriter& writer, DiagnosticLevel level, const std::string& owner, const std::string& summary,
                    const KeyValueFields& fields = {}) -> void;
auto EmitArtifact(SchemaWriter& writer, const std::string& label, const std::filesystem::path& path, const std::string& detail = {})
    -> void;

}  // namespace icts
