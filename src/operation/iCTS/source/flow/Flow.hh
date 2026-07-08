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
 * @file Flow.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS flow lifecycle owner.
 */

#pragma once

#include <string>

#include "database/adapter/fast_sta/FastSta.hh"
#include "database/config/Config.hh"
#include "database/design/Design.hh"
#include "database/io/Wrapper.hh"
#include "design/ClockLayout.hh"
#include "evaluation/Evaluation.hh"
#include "evaluation/qor/QorEvaluation.hh"
#include "instantiation/Instantiation.hh"
#include "optimization/Optimization.hh"
#include "synthesis/htree/characterization/library/CharacterizationLibrary.hh"
#include "synthesis/trace/SynthesisTrace.hh"
#include "utils/logger/Schema.hh"

namespace icts {

enum class FlowRunStatusCode
{
  kFinished,
  kNoOp,
  kSetupNotReady,
  kReadDataFailed,
  kSynthesisFailed,
  kInstantiationFailed
};

struct FlowRunStatus
{
  FlowRunStatusCode code = FlowRunStatusCode::kSetupNotReady;
  std::string step;
  std::string message;

  auto ok() const -> bool { return code == FlowRunStatusCode::kFinished || code == FlowRunStatusCode::kNoOp; }
};

enum class FlowReportStatusCode
{
  kFinished,
  kFailed
};

struct FlowReportStatus
{
  FlowReportStatusCode code = FlowReportStatusCode::kFailed;
  std::string message;

  auto ok() const -> bool { return code == FlowReportStatusCode::kFinished; }
};

struct CTSRuntime
{
  Config config;
  Design design;
  Wrapper wrapper;
  FastSTA fast_sta;
  SchemaWriter reporter;

  auto reset() -> void
  {
    config.reset();
    design.reset();
    wrapper.reset();
    fast_sta.reset();
    reporter.reset();
  }
};

class Flow
{
 public:
  explicit Flow(CTSRuntime& runtime) : _runtime(runtime) {}
  ~Flow() = default;

  auto runCTS() -> FlowRunStatus;
  auto emitReports(const std::string& save_dir) -> FlowReportStatus;
  auto outputRuntimeSetup() -> void;
  auto outputSummary() const -> QorSummary;
  auto outputRunSummary() const -> SynthesisTraceSummary;
  auto setSetupReady(bool setup_ready) -> void { _setup_ready = setup_ready; }
  auto reset() -> void;

  Flow(const Flow& other) = delete;
  Flow(Flow&& other) = delete;
  auto operator=(const Flow& other) -> Flow& = delete;
  auto operator=(Flow&& other) -> Flow& = delete;

 private:
  struct ClockDataReadSummary
  {
    std::string reason;
    bool success = false;
  };

  auto readClockData() -> ClockDataReadSummary;
  auto runSynthesis() -> SynthesisTraceSummary;
  auto runOptimization() -> OptimizationSummary;
  auto instantiateClockTree() -> InstantiationSummary;
  auto evaluateClockTree() -> EvaluationBuild;
  auto emitKeyResults(double elapsed_time_s, double peak_vmem_delta_mb) const -> void;

  CTSRuntime& _runtime;
  SynthesisTraceSummary _run_summary;
  ClockLayout _clock_layout;
  EvaluationState _evaluation_state;
  InstantiationSummary _instantiation_summary;
  CharacterizationLibrary _char_library;
  bool _runtime_setup_emitted = false;
  bool _setup_ready = false;
  bool _evaluation_ready = false;
};

}  // namespace icts
