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
 * @file Evaluation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS evaluation entry facade implementation.
 */

#include "evaluation/Evaluation.hh"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>

#include "Log.hh"
#include "logger/Schema.hh"

namespace icts {

auto Evaluation::run(EvaluationState evaluation_state, const EvaluationInput& input) -> EvaluationBuild
{
  LOG_FATAL_IF(input.reporter == nullptr) << "Evaluation requires reporter.";
  auto& reporter = *input.reporter;
  auto runtime = reporter.beginRuntimeMetric("evaluation");
  auto evaluation_stage
      = reporter.beginStage("Evaluation", "Evaluate CTS clock tree", {}, StageReportOptions{.emit_success_summary = false});
  reporter.emitSection("## Evaluation Overview");
  reporter.emitSection("### Final Evaluation");
  evaluate(evaluation_state, input);
  const bool evaluation_ready = isEvaluationReady(evaluation_state);
  if (evaluation_ready) {
    (void) runtime.finished();
    evaluation_stage.finished();
  } else {
    (void) runtime.failed();
    evaluation_stage.failed();
  }
  return EvaluationBuild{
      .output = EvaluationOutput{.state = std::move(evaluation_state)},
      .summary = EvaluationSummary{.evaluation_ready = evaluation_ready, .status = evaluation_ready ? "finished" : "failed"},
  };
}

auto Evaluation::evaluate(EvaluationState& state, const EvaluationInput& input) -> void
{
  QorEvaluation::evaluate(state, input);
}

auto Evaluation::outputSummary(const EvaluationState& state) -> QorSummary
{
  return QorEvaluation::outputSummary(state);
}

auto Evaluation::isEvaluationReady(const EvaluationState& state) -> bool
{
  return QorEvaluation::isEvaluationReady(state);
}

auto Evaluation::reset(EvaluationState& state) -> void
{
  QorEvaluation::reset(state);
}

}  // namespace icts
