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
 * @file SchemaScope.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Runtime metric and stage scope helpers for the structured report writer.
 */

#include <glog/logging.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "LogFormat.hh"
#include "Schema.hh"
#include "usage.hh"

namespace icts {
namespace {

constexpr const char* kStatusFinished = "finished";
constexpr const char* kStatusFailed = "failed";
constexpr const char* kStatusSkipped = "skipped";

auto NormalizeStatus(const std::string& status) -> std::string
{
  if (status == "success") {
    return kStatusFinished;
  }
  return status;
}

auto StageMarkerForStatus(const std::string& status) -> std::string
{
  if (status == kStatusFailed) {
    return "FAILED";
  }
  if (status == kStatusSkipped) {
    return "SKIPPED";
  }
  return "FINISHED";
}

auto FormatSeconds(std::chrono::steady_clock::duration duration) -> std::string
{
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  const double seconds = static_cast<double>(milliseconds) / 1000.0;

  std::ostringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream.precision(3);
  stream << seconds;
  return stream.str();
}

auto FlattenFields(const KeyValueFields& fields) -> std::string
{
  std::ostringstream stream;
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0U) {
      stream << ", ";
    }
    stream << fields.at(index).first << "=" << fields.at(index).second;
  }
  return stream.str();
}

}  // namespace

SchemaWriter::RuntimeMetricScope::RuntimeMetricScope(SchemaWriter& writer, std::string stage)
    : _writer(&writer), _stage(std::move(stage)), _stats(std::make_unique<ieda::Stats>())
{
}

SchemaWriter::RuntimeMetricScope::RuntimeMetricScope(RuntimeMetricScope&& other) noexcept
    : _writer(other._writer), _stage(std::move(other._stage)), _stats(std::move(other._stats)), _finished(other._finished)
{
  other._writer = nullptr;
  other._finished = true;
}

auto SchemaWriter::RuntimeMetricScope::operator=(RuntimeMetricScope&& other) noexcept -> RuntimeMetricScope&
{
  if (this == &other) {
    return *this;
  }
  _writer = other._writer;
  _stage = std::move(other._stage);
  _stats = std::move(other._stats);
  _finished = other._finished;
  other._writer = nullptr;
  other._finished = true;
  return *this;
}

SchemaWriter::RuntimeMetricScope::~RuntimeMetricScope() = default;

auto SchemaWriter::RuntimeMetricScope::finish(const std::string& status) -> RuntimeMetricRecord
{
  const auto metric_record = measure();
  if (!_finished && _writer != nullptr) {
    _writer->recordRuntimeMetric(_stage, NormalizeStatus(status), metric_record);
    _finished = true;
  }
  return metric_record;
}

auto SchemaWriter::RuntimeMetricScope::finished() -> RuntimeMetricRecord
{
  return finish(kStatusFinished);
}

auto SchemaWriter::RuntimeMetricScope::failed() -> RuntimeMetricRecord
{
  return finish(kStatusFailed);
}

auto SchemaWriter::RuntimeMetricScope::measure() const -> RuntimeMetricRecord
{
  if (_stats == nullptr) {
    return {};
  }
  return RuntimeMetricRecord{
      .elapsed_time_s = _stats->elapsedRunTime(),
      .peak_vmem_delta_mb = _stats->memoryDelta(),
  };
}

auto SchemaWriter::resetRuntimeMetrics() -> void
{
  const std::scoped_lock lock(_mutex);
  _runtime_metrics.clear();
}

auto SchemaWriter::beginRuntimeMetric(std::string stage) -> RuntimeMetricScope
{
  return RuntimeMetricScope(*this, std::move(stage));
}

auto SchemaWriter::recordRuntimeMetric(std::string stage, std::string status, const RuntimeMetricRecord& metric_record) -> void
{
  const std::scoped_lock lock(_mutex);
  _runtime_metrics.push_back(RuntimeMetric{
      .stage = std::move(stage),
      .status = std::move(status),
      .elapsed_time_s = metric_record.elapsed_time_s,
      .peak_vmem_delta_mb = metric_record.peak_vmem_delta_mb,
  });
}

auto SchemaWriter::emitRuntimeSummary(const std::string& title) -> void
{
  TableRows rows;
  {
    const std::scoped_lock lock(_mutex);
    rows.reserve(_runtime_metrics.size());
    for (const auto& metric : _runtime_metrics) {
      rows.push_back({
          metric.stage,
          metric.status,
          logformat::FormatFixed(metric.elapsed_time_s, 3),
          logformat::FormatFixed(metric.peak_vmem_delta_mb, 3),
      });
    }
  }
  if (!rows.empty()) {
    EmitTable(*this, title, {"Stage", "Status", "Elapsed Time (s)", "Peak VMem Delta (MB)"}, rows);
  }
}

auto SchemaWriter::emitRuntimeMetricTable(const std::string& title, const std::string& stage, const std::string& status,
                                          const RuntimeMetricRecord& metric_record) -> void
{
  const std::string normalized_status = NormalizeStatus(status);
  const std::vector<std::string> headers = {"Stage", "Status", "Elapsed Time (s)", "Peak VMem Delta (MB)"};
  const TableRows rows = {
      {stage, normalized_status, logformat::FormatFixed(metric_record.elapsed_time_s, 3),
       logformat::FormatFixed(metric_record.peak_vmem_delta_mb, 3)},
  };

  LOG_INFO << "";
  LOG_INFO << logformat::MakeTitledTable(title, headers, rows);
  emitTable(title, headers, rows);
}

auto SchemaWriter::beginStage(std::string module, std::string stage, const KeyValueFields& start_fields) -> StageScope
{
  return beginStage(std::move(module), std::move(stage), start_fields, StageReportOptions{});
}

auto SchemaWriter::beginStage(std::string module, std::string stage, const KeyValueFields& start_fields, StageReportOptions report_options)
    -> StageScope
{
  return StageScope(*this, std::move(module), std::move(stage), start_fields, report_options);
}

SchemaWriter::StageScope::StageScope(SchemaWriter& writer, std::string module, std::string stage, const KeyValueFields& start_fields,
                                     StageReportOptions report_options)
    : _module(std::move(module)),
      _stage(std::move(stage)),
      _report_options(report_options),
      _writer(&writer),
      _start_time(std::chrono::steady_clock::now())
{
  LOG_INFO << "";
  LOG_INFO << logformat::MakeStageMarker(_module, _stage, "START");
  if (!start_fields.empty()) {
    _writer->emitKeyValueTableTo(_module + " " + _stage + " Context", start_fields, _report_options.context_sink);
  }
}

SchemaWriter::StageScope::StageScope(StageScope&& other) noexcept
    : _module(std::move(other._module)),
      _stage(std::move(other._stage)),
      _report_options(other._report_options),
      _writer(other._writer),
      _start_time(other._start_time),
      _finished(other._finished)
{
  other._writer = nullptr;
  other._finished = true;
}

auto SchemaWriter::StageScope::operator=(StageScope&& other) noexcept -> StageScope&
{
  if (this == &other) {
    return *this;
  }
  if (!_finished) {
    finished();
  }
  _module = std::move(other._module);
  _stage = std::move(other._stage);
  _report_options = other._report_options;
  _writer = other._writer;
  _start_time = other._start_time;
  _finished = other._finished;
  other._writer = nullptr;
  other._finished = true;
  return *this;
}

SchemaWriter::StageScope::~StageScope()
{
  if (!_finished) {
    finished();
  }
}

auto SchemaWriter::StageScope::markRunning(const std::string& summary, const KeyValueFields& fields) -> void
{
  if (!summary.empty()) {
    LOG_INFO << logformat::MakeStageMarker(_module, summary, "RUNNING");
  } else {
    LOG_INFO << logformat::MakeStageMarker(_module, _stage, "RUNNING");
  }
  (void) fields;
}

auto SchemaWriter::StageScope::finished(const KeyValueFields& finish_fields) -> void
{
  closeWithStatus(kStatusFinished, finish_fields);
}

auto SchemaWriter::StageScope::failed(const KeyValueFields& finish_fields) -> void
{
  closeWithStatus(kStatusFailed, finish_fields);
}

auto SchemaWriter::StageScope::skip(const KeyValueFields& finish_fields) -> void
{
  closeWithStatus(kStatusSkipped, finish_fields);
}

auto SchemaWriter::StageScope::closeWithStatus(const std::string& status, const KeyValueFields& finish_fields) -> void
{
  if (_finished) {
    return;
  }
  _finished = true;

  KeyValueFields schema_fields;
  schema_fields.reserve(finish_fields.size() + 1U);
  schema_fields.emplace_back("status", status);
  schema_fields.insert(schema_fields.end(), finish_fields.begin(), finish_fields.end());

  KeyValueFields console_fields = schema_fields;
  console_fields.emplace_back("elapsed_time", FormatSeconds(std::chrono::steady_clock::now() - _start_time) + " s");
  const std::string console_summary = FlattenFields(console_fields);
  LOG_INFO << logformat::MakeStageMarker(_module, _stage, StageMarkerForStatus(status))
           << (console_summary.empty() ? std::string{} : ": " + console_summary);
  if (_writer != nullptr) {
    auto summary_sink = _report_options.summary_sink;
    if (status == kStatusFinished && !_report_options.emit_success_summary) {
      summary_sink = ReportSink::kNone;
    } else if (status != kStatusFinished) {
      if (summary_sink == ReportSink::kNone) {
        summary_sink = ReportSink::kDefault;
      } else if (summary_sink == ReportSink::kDetail) {
        summary_sink = ReportSink::kBoth;
      }
    }
    _writer->emitKeyValueTableTo(_module + " " + _stage + " Summary", schema_fields, summary_sink);
  }
}

}  // namespace icts
