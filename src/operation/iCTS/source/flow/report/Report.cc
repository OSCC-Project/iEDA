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
 * @file Report.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS report entry facade implementation.
 */

#include "report/Report.hh"

#include <glog/logging.h>

#include <filesystem>
#include <ostream>
#include <utility>

#include "Log.hh"
#include "config/Config.hh"
#include "evaluation/Evaluation.hh"
#include "evaluation/qor/QorEvaluation.hh"
#include "logger/Schema.hh"
#include "report/export/ReportExport.hh"
#include "report/overview/Overview.hh"
#include "report/qor/QorReport.hh"
#include "report/visualization/Visualization.hh"

namespace icts {

auto Report::run(const ReportInput& input) -> ReportSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "Report requires config.";
  LOG_FATAL_IF(input.design == nullptr) << "Report requires design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Report requires wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Report requires reporter.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "Report requires clock layout.";
  LOG_FATAL_IF(input.evaluation_state == nullptr) << "Report requires evaluation state.";
  const auto& config = *input.config;
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;
  auto& evaluation_state = *input.evaluation_state;
  LOG_FATAL_IF(config.get_work_dir().empty()) << "CTS report requires initialized CTS run setup.";

  auto runtime = reporter.beginRuntimeMetric("report");
  auto report_stage = reporter.beginStage("Report", "Emit CTS statistics and visualization reports", {},
                                          StageReportOptions{.emit_success_summary = false});
  const auto paths = ReportExport::resolvePaths(config, input.save_dir);
  const bool reused_evaluation_state = input.evaluation_ready && Evaluation::isEvaluationReady(evaluation_state);

  reporter.emitSection("## Report Overview");
  Overview::emitReportMode(reporter, reused_evaluation_state, paths);

  bool current_evaluation_ready = input.evaluation_ready;
  if (!reused_evaluation_state) {
    reporter.emitSection("### Report Evaluation");
    const auto evaluation_output = Evaluation::run(
        evaluation_state,
        EvaluationInput{
            .config = &config, .clock_layout = input.clock_layout, .design = &design, .wrapper = &wrapper, .reporter = &reporter});
    evaluation_state = evaluation_output.output.state;
    current_evaluation_ready = evaluation_output.summary.evaluation_ready;
  }

  const bool statistics_success
      = current_evaluation_ready && QorReport::write(reporter, evaluation_state, paths.statistics_dir.string(), false);
  const auto visualization_summary = Visualization::emit(VisualizationInput{.config = &config,
                                                                            .design = &design,
                                                                            .wrapper = &wrapper,
                                                                            .reporter = &reporter,
                                                                            .visualization_dir = paths.visualization_dir,
                                                                            .clock_layout = input.clock_layout},
                                                         VisualizationConfig{});
  const bool report_success = statistics_success && visualization_summary.success;
  const auto report_metric = report_success ? runtime.finished() : runtime.failed();
  reporter.emitRuntimeMetricTable("CTS Report Runtime", "report", report_success ? "finished" : "failed", report_metric);
  if (report_success) {
    report_stage.finished({{"statistics_status", statistics_success ? "finished" : "failed"},
                           {"visualization_status", visualization_summary.svg_success ? "finished" : "failed"},
                           {"gds_status", visualization_summary.gds_success ? "finished" : "failed"}});
  } else {
    report_stage.failed({{"statistics_status", statistics_success ? "finished" : "failed"},
                         {"visualization_status", visualization_summary.svg_success ? "finished" : "failed"},
                         {"gds_status", visualization_summary.gds_success ? "finished" : "failed"}});
  }

  return ReportSummary{
      .success = report_success,
      .evaluation_ready = current_evaluation_ready,
      .statistics_success = statistics_success,
      .svg_success = visualization_summary.svg_success,
      .gds_success = visualization_summary.gds_success,
  };
}

}  // namespace icts
