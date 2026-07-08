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
 * @file Flow.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS flow lifecycle owner implementation.
 */

#include "Flow.hh"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ClockDAG.hh"
#include "ClockLayout.hh"
#include "design/Design.hh"
#include "evaluation/Evaluation.hh"
#include "instantiation/Instantiation.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"
#include "optimization/Optimization.hh"
#include "report/Report.hh"
#include "setup/Setup.hh"
#include "setup/clock_data/ClockDataRead.hh"
#include "synthesis/Synthesis.hh"

namespace icts {
namespace {

auto formatValueWithUnit(const std::string& value, const std::string& unit) -> std::string
{
  if (value == "n/a" || unit.empty()) {
    return value;
  }
  return value + " " + unit;
}

auto formatOptionalCount(std::size_t value) -> std::string
{
  return value == 0U ? std::string{"n/a"} : std::to_string(value);
}

auto formatOptionalUnsigned(unsigned value) -> std::string
{
  return value == 0U ? std::string{"n/a"} : std::to_string(value);
}

auto synthesisOutcomeName(SynthesisOutcome outcome) -> std::string
{
  switch (outcome) {
    case SynthesisOutcome::kFinished:
      return "finished";
    case SynthesisOutcome::kFailed:
      return "failed";
    case SynthesisOutcome::kNoOp:
      return "no_op";
  }
  return "failed";
}

auto buildCompletedRunStatus(const SynthesisTraceSummary& run_summary, const InstantiationSummary& instantiation_summary) -> FlowRunStatus
{
  if (run_summary.outcome == SynthesisOutcome::kFinished && run_summary.success && instantiation_summary.success) {
    return FlowRunStatus{.code = FlowRunStatusCode::kFinished, .step = "flow", .message = "CTS flow finished."};
  }
  if (run_summary.outcome == SynthesisOutcome::kNoOp) {
    return FlowRunStatus{.code = FlowRunStatusCode::kNoOp, .step = "synthesis", .message = run_summary.no_op_reason};
  }
  if (run_summary.outcome == SynthesisOutcome::kFinished && !instantiation_summary.success) {
    return FlowRunStatus{.code = FlowRunStatusCode::kInstantiationFailed, .step = "instantiation", .message = "CTS instantiation failed."};
  }
  return FlowRunStatus{.code = FlowRunStatusCode::kSynthesisFailed, .step = "synthesis", .message = "CTS synthesis flow failed."};
}

}  // namespace

auto Flow::runCTS() -> FlowRunStatus
{
  _runtime.reporter.resetRuntimeMetrics();
  auto total_runtime = _runtime.reporter.beginRuntimeMetric("total");
  auto run_stage
      = _runtime.reporter.beginStage("CTS", "Clock tree synthesis API flow", {}, StageReportOptions{.emit_success_summary = false});

  if (!_setup_ready) {
    _run_summary = SynthesisTraceSummary{};
    _run_summary.success = false;
    _run_summary.outcome = SynthesisOutcome::kFailed;
    run_stage.failed({{"reason", "setup_failed"}});
    const auto total_metric = total_runtime.failed();
    _runtime.reporter.emitSection("## Runtime Overview");
    _runtime.reporter.emitRuntimeSummary("CTS Runtime Overview");
    emitKeyResults(total_metric.elapsed_time_s, total_metric.peak_vmem_delta_mb);
    return FlowRunStatus{.code = FlowRunStatusCode::kSetupNotReady, .step = "setup", .message = "CTS setup is not ready."};
  }

  const auto clock_data_read_summary = readClockData();
  if (!clock_data_read_summary.success) {
    _run_summary = SynthesisTraceSummary{};
    _run_summary.success = false;
    _run_summary.outcome = SynthesisOutcome::kFailed;
    run_stage.failed({{"reason", clock_data_read_summary.reason}});
    const auto total_metric = total_runtime.failed();
    _runtime.reporter.emitSection("## Runtime Overview");
    _runtime.reporter.emitRuntimeSummary("CTS Runtime Overview");
    emitKeyResults(total_metric.elapsed_time_s, total_metric.peak_vmem_delta_mb);
    return FlowRunStatus{.code = FlowRunStatusCode::kReadDataFailed, .step = "read_data", .message = clock_data_read_summary.reason};
  }
  (void) runSynthesis();
  (void) runOptimization();
  (void) instantiateClockTree();
  (void) evaluateClockTree();

  const bool run_success = _run_summary.outcome == SynthesisOutcome::kFinished && _run_summary.success && _instantiation_summary.success;
  const bool run_no_op = _run_summary.outcome == SynthesisOutcome::kNoOp;
  SchemaWriter::RuntimeMetricRecord total_metric;
  if (run_success) {
    total_metric = total_runtime.finished();
  } else if (run_no_op) {
    total_metric = total_runtime.finish("no_op");
  } else {
    total_metric = total_runtime.failed();
  }
  run_stage.markRunning("Main CTS flow finished");
  if (run_success) {
    run_stage.finished();
  } else if (run_no_op) {
    run_stage.skip({{"reason", _run_summary.no_op_reason}});
  } else {
    run_stage.failed();
  }

  _runtime.reporter.emitSection("## Runtime Overview");
  _runtime.reporter.emitRuntimeSummary("CTS Runtime Overview");
  emitKeyResults(total_metric.elapsed_time_s, total_metric.peak_vmem_delta_mb);
  return buildCompletedRunStatus(_run_summary, _instantiation_summary);
}

auto Flow::readClockData() -> Flow::ClockDataReadSummary
{
  _run_summary = SynthesisTraceSummary{};
  _clock_layout.reset();
  _char_library = CharacterizationLibrary{};
  Evaluation::reset(_evaluation_state);
  _instantiation_summary = InstantiationSummary{};
  _evaluation_ready = false;

  auto runtime = _runtime.reporter.beginRuntimeMetric("read_data");
  auto read_stage
      = _runtime.reporter.beginStage("CTSReadData", "Read CTS clock data", {}, StageReportOptions{.emit_success_summary = false});
  _runtime.reporter.emitSection("## CTS Clock Data Overview");
  _runtime.reporter.emitSection("### Clock Data");
  const bool read_data_ready = ClockDataRead::read(ClockDataReadInput{
      .config = &_runtime.config,
      .design = &_runtime.design,
      .wrapper = &_runtime.wrapper,
      .reporter = &_runtime.reporter,
  });
  if (read_data_ready) {
    (void) runtime.finished();
    read_stage.finished();
    return ClockDataReadSummary{.reason = "n/a", .success = true};
  } else {
    (void) runtime.failed();
    read_stage.failed({{"reason", "sdc_clock_resolution_failed"}});
    return ClockDataReadSummary{.reason = "read_data_failed", .success = false};
  }
}

auto Flow::runSynthesis() -> SynthesisTraceSummary
{
  Evaluation::reset(_evaluation_state);
  _evaluation_ready = false;
  _instantiation_summary = InstantiationSummary{};
  _runtime.design.clearClockDAG();
  _run_summary = Synthesis::run(SynthesisInput{
      .config = &_runtime.config,
      .design = &_runtime.design,
      .wrapper = &_runtime.wrapper,
      .fast_sta = &_runtime.fast_sta,
      .reporter = &_runtime.reporter,
      .clock_layout = &_clock_layout,
      .characterization_library = &_char_library,
  });
  if (_run_summary.outcome == SynthesisOutcome::kFinished && _run_summary.success && !_runtime.design.rebuildClockDAG()) {
    _run_summary.success = false;
    _run_summary.outcome = SynthesisOutcome::kFailed;
    EmitDiagnostic(_runtime.reporter, DiagnosticLevel::kError, "CTSFlow",
                   "synthesized CTS topology is not a valid clock DAG; instantiation and final evaluation are blocked.",
                   {{"reason", _runtime.design.get_clock_dag().get_status()}});
    return _run_summary;
  }
  return _run_summary;
}

auto Flow::runOptimization() -> OptimizationSummary
{
  if (_run_summary.outcome != SynthesisOutcome::kFinished || !_run_summary.success) {
    return OptimizationSummary{};
  }
  const auto optimization_summary = Optimization::run(OptimizationInput{.config = &_runtime.config,
                                                                        .design = &_runtime.design,
                                                                        .wrapper = &_runtime.wrapper,
                                                                        .fast_sta = &_runtime.fast_sta,
                                                                        .reporter = &_runtime.reporter,
                                                                        .clock_layout = &_clock_layout,
                                                                        .characterization_library = &_char_library});
  if (!optimization_summary.success) {
    _run_summary.success = false;
    _run_summary.outcome = SynthesisOutcome::kFailed;
    return optimization_summary;
  }
  if (!_runtime.design.rebuildClockDAG()) {
    _run_summary.success = false;
    _run_summary.outcome = SynthesisOutcome::kFailed;
    EmitDiagnostic(_runtime.reporter, DiagnosticLevel::kError, "CTSFlow",
                   "optimized CTS topology is not a valid clock DAG; instantiation and final evaluation are blocked.",
                   {{"reason", _runtime.design.get_clock_dag().get_status()}});
  }
  return optimization_summary;
}

auto Flow::instantiateClockTree() -> InstantiationSummary
{
  _instantiation_summary = InstantiationSummary{};
  if (_run_summary.outcome == SynthesisOutcome::kFinished && _run_summary.success) {
    _instantiation_summary = Instantiation::run(InstantiationInput{
        .design = &_runtime.design,
        .wrapper = &_runtime.wrapper,
        .reporter = &_runtime.reporter,
    });
    _clock_layout.markInstantiationDone(_instantiation_summary.success);
    _run_summary.success = _instantiation_summary.success;
  }
  return _instantiation_summary;
}

auto Flow::evaluateClockTree() -> EvaluationBuild
{
  if (_run_summary.outcome != SynthesisOutcome::kFinished || !_instantiation_summary.success) {
    Evaluation::reset(_evaluation_state);
    _evaluation_ready = false;
    return EvaluationBuild{.output = EvaluationOutput{.state = _evaluation_state}, .summary = EvaluationSummary{.evaluation_ready = false}};
  }
  const auto output = Evaluation::run(_evaluation_state, EvaluationInput{
                                                             .config = &_runtime.config,
                                                             .clock_layout = &_clock_layout,
                                                             .design = &_runtime.design,
                                                             .wrapper = &_runtime.wrapper,
                                                             .reporter = &_runtime.reporter,
                                                         });
  _evaluation_state = output.output.state;
  _evaluation_ready = output.summary.evaluation_ready;
  return output;
}

auto Flow::emitReports(const std::string& save_dir) -> FlowReportStatus
{
  const auto report_summary = Report::run(ReportInput{.config = &_runtime.config,
                                                      .design = &_runtime.design,
                                                      .wrapper = &_runtime.wrapper,
                                                      .reporter = &_runtime.reporter,
                                                      .save_dir = save_dir,
                                                      .evaluation_ready = _evaluation_ready,
                                                      .clock_layout = &_clock_layout,
                                                      .evaluation_state = &_evaluation_state});
  _evaluation_ready = report_summary.evaluation_ready;
  if (report_summary.success) {
    return FlowReportStatus{.code = FlowReportStatusCode::kFinished, .message = "CTS reports emitted."};
  }
  return FlowReportStatus{.code = FlowReportStatusCode::kFailed, .message = "CTS report generation failed."};
}

auto Flow::outputRuntimeSetup() -> void
{
  if (_runtime_setup_emitted) {
    return;
  }
  _runtime_setup_emitted = true;

  Setup::emitRuntimeSetup(RuntimeSetupInput{
      .config = &_runtime.config,
      .wrapper = &_runtime.wrapper,
      .reporter = &_runtime.reporter,
  });
}

auto Flow::emitKeyResults(double elapsed_time_s, double peak_vmem_delta_mb) const -> void
{
  const auto evaluation_summary = outputSummary();
  const std::size_t sink_count = _run_summary.hard_macro_sinks + _run_summary.regular_sinks;

  KeyValueFields fields = {
      {"status", synthesisOutcomeName(_run_summary.outcome)},
  };
  if (!_run_summary.no_op_reason.empty()) {
    fields.emplace_back("no_op_reason", _run_summary.no_op_reason);
  }
  fields.insert(fields.end(),
                {
                    {"clock_count", std::to_string(_run_summary.total_clocks)},
                    {"sink_count", std::to_string(sink_count)},
                    {"sink_domain_count", std::to_string(_run_summary.total_sink_domains)},
                    {"selected_htree_level_count", formatOptionalCount(_run_summary.selected_htree_level_count)},
                    {"selected_htree_depth", formatOptionalUnsigned(_run_summary.selected_htree_depth)},
                    {"htree_inserted_buffer_count", std::to_string(_run_summary.htree_inserted_buffer_count)},
                    {"final_clock_buffer_count", std::to_string(evaluation_summary.final_clock_buffer_count)},
                    {"final_buffer_area", formatValueWithUnit(logformat::FormatFixed(evaluation_summary.final_buffer_area_um2, 3), "um^2")},
                    {"max_clock_net_wirelength",
                     formatValueWithUnit(logformat::FormatFixed(evaluation_summary.max_clock_net_wirelength_um, 3), "um")},
                    {"total_clock_network_wirelength",
                     formatValueWithUnit(logformat::FormatFixed(evaluation_summary.total_clock_network_wirelength_um, 3), "um")},
                    {"elapsed_time", formatValueWithUnit(logformat::FormatFixed(elapsed_time_s, 3), "s")},
                    {"peak_vmem_delta", formatValueWithUnit(logformat::FormatFixed(peak_vmem_delta_mb, 3), "MB")},
                });

  _runtime.reporter.emitSection("## Run Results");
  EmitKeyValueTable(_runtime.reporter, "CTS Key Results", fields);
}

auto Flow::outputSummary() const -> QorSummary
{
  if (!_evaluation_ready) {
    return {};
  }
  return Evaluation::outputSummary(_evaluation_state);
}

auto Flow::outputRunSummary() const -> SynthesisTraceSummary
{
  return _run_summary;
}

auto Flow::reset() -> void
{
  Evaluation::reset(_evaluation_state);
  _run_summary = SynthesisTraceSummary{};
  _clock_layout.reset();
  _char_library = CharacterizationLibrary{};
  _instantiation_summary = InstantiationSummary{};
  _runtime_setup_emitted = false;
  _setup_ready = false;
  _evaluation_ready = false;
}

}  // namespace icts
