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
 * @file Optimization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS post-synthesis optimization flow facade implementation.
 */

#include "optimization/Optimization.hh"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "FastSta.hh"
#include "Log.hh"
#include "config/Config.hh"
#include "design/Clock.hh"
#include "design/Design.hh"
#include "io/Wrapper.hh"
#include "logger/LogFormat.hh"
#include "logger/Schema.hh"
#include "optimization/clock_sizing_edit/ClockSizingAcceptedEdit.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"
#include "optimization/policy/OptimizationPolicy.hh"
#include "optimization/preparation/OptimizationPreparation.hh"
#include "optimization/report/OptimizationReport.hh"
#include "optimization/solver/OptimizationSolver.hh"

namespace icts {
namespace oi = clock_sizing_optimization;
namespace {

class FastStaClockContextGuard
{
 public:
  FastStaClockContextGuard(FastSTA& fast_sta, FastStaClockId clock_id) : _fast_sta(&fast_sta), _clock_id(clock_id) {}
  ~FastStaClockContextGuard() { erase(); }

  FastStaClockContextGuard(const FastStaClockContextGuard&) = delete;
  auto operator=(const FastStaClockContextGuard&) -> FastStaClockContextGuard& = delete;

  FastStaClockContextGuard(FastStaClockContextGuard&& rhs) noexcept : _fast_sta(rhs._fast_sta), _clock_id(rhs._clock_id)
  {
    rhs._fast_sta = nullptr;
    rhs._clock_id = kInvalidFastStaClockId;
  }

  auto operator=(FastStaClockContextGuard&& rhs) noexcept -> FastStaClockContextGuard&
  {
    if (this == &rhs) {
      return *this;
    }
    erase();
    _fast_sta = rhs._fast_sta;
    _clock_id = rhs._clock_id;
    rhs._fast_sta = nullptr;
    rhs._clock_id = kInvalidFastStaClockId;
    return *this;
  }

  auto id() const -> FastStaClockId { return _clock_id; }

 private:
  auto erase() -> void
  {
    if (_fast_sta != nullptr && _clock_id != kInvalidFastStaClockId) {
      (void) _fast_sta->eraseClockContext(_clock_id);
      _clock_id = kInvalidFastStaClockId;
    }
  }

  FastSTA* _fast_sta = nullptr;
  FastStaClockId _clock_id = kInvalidFastStaClockId;
};

auto resolveRoutingLayer(const Config& config) -> int
{
  const auto& routing_layers = config.get_routing_layers();
  LOG_FATAL_IF(routing_layers.empty() || routing_layers.front() == 0U)
      << "Optimization: routing layer must be configured before FastSTA context construction.";
  return static_cast<int>(routing_layers.front());
}

auto resolveWireWidth(const Config& config) -> std::optional<double>
{
  const double wire_width_um = config.get_wire_width();
  return wire_width_um > 0.0 ? std::optional<double>{wire_width_um} : std::nullopt;
}

auto buildFastStaEnvironment(const OptimizationInput& input) -> FastStaEnvironment
{
  const auto& config = *input.config;
  auto& wrapper = *input.wrapper;
  const auto dbu_per_um = wrapper.queryDbUnit();
  LOG_FATAL_IF(dbu_per_um <= 0) << "Optimization: DBU-per-micron is unavailable before FastSTA context construction.";
  return FastStaEnvironment{
      .wrapper = &wrapper,
      .dbu_per_um = dbu_per_um,
      .routing_layer = resolveRoutingLayer(config),
      .wire_width_um = resolveWireWidth(config),
      .root_input_slew_ns = std::max(0.0, config.get_root_input_slew()),
      .max_cap_pf = config.has_max_cap() && config.get_max_cap() > 0.0 ? std::optional<double>{config.get_max_cap()} : std::nullopt,
      .max_sink_tran_ns = config.get_max_sink_tran(),
  };
}

}  // namespace

auto Optimization::run(const OptimizationInput& input) -> OptimizationSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "Optimization requires config.";
  LOG_FATAL_IF(input.design == nullptr) << "Optimization requires design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Optimization requires wrapper.";
  LOG_FATAL_IF(input.fast_sta == nullptr) << "Optimization requires FastSTA.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Optimization requires reporter.";
  LOG_FATAL_IF(input.clock_layout == nullptr) << "Optimization requires clock layout.";
  LOG_FATAL_IF(input.characterization_library == nullptr) << "Optimization requires characterization library.";
  const auto& config = *input.config;
  auto& design = *input.design;
  auto& wrapper = *input.wrapper;
  auto& fast_sta = *input.fast_sta;
  auto& reporter = *input.reporter;
  auto& clock_layout = *input.clock_layout;
  auto& characterization_library = *input.characterization_library;
  (void) characterization_library;
  OptimizationSummary optimization_summary;
  auto runtime = reporter.beginRuntimeMetric("optimization");
  auto stage = reporter.beginStage("Optimization", "Optimize synthesized CTS buffers with CTS fast STA", {},
                                   StageReportOptions{.emit_success_summary = false});
  reporter.emitSection("## Optimization Overview");

  const auto& policy = oi::DefaultOptimizationPolicy();
  if (!oi::ValidateOptimizationPolicy(policy)) {
    LOG_ERROR << "Optimization: internal optimizer policy is invalid.";
    optimization_summary.success = false;
    optimization_summary.status = "failed";
    optimization_summary.reason = "invalid_optimizer_options";
    (void) runtime.failed();
    stage.failed({{"reason", "invalid_optimizer_options"}});
    return optimization_summary;
  }

  const auto start_time = std::chrono::steady_clock::now();
  const auto clocks = design.get_clocks();
  optimization_summary.clock_count = clocks.size();
  fast_sta.bindEnvironment(buildFastStaEnvironment(input));
  const auto master_infos = oi::CollectClockSizingBufferMasters(oi::ClockSizingMasterQueryInput{
      .wrapper = &wrapper,
      .buffer_cell_masters = &config.get_buffer_types(),
  });
  if (master_infos.empty()) {
    LOG_WARNING << "Optimization: skip because no legal buffer sizing candidates are available.";
    optimization_summary.reason = "no_sizing_candidates";
    (void) runtime.finish("skipped");
    stage.skip({{"reason", "no_sizing_candidates"}});
    return optimization_summary;
  }

  auto stage_start = std::chrono::steady_clock::now();
  const auto route_tree_by_net = oi::BuildClockSizingRouteTrees(design, clocks);
  const double route_tree_cache_runtime_s = oi::ElapsedSeconds(stage_start);
  const double target_skew_ns = std::max(0.0, config.get_skew_bound());
  EmitKeyValueTable(reporter, "CTS Optimization Setup",
                    {{"timing_source", "cts_fast_sta_incremental"},
                     {"target_skew", oi::FormatNs(target_skew_ns)},
                     {"candidate_master_count", std::to_string(master_infos.size())}});
  EmitKeyValueTable(reporter, "CTS Optimization Global Profile",
                    {{"build_route_tree_cache", oi::FormatSeconds(route_tree_cache_runtime_s)},
                     {"cached_route_tree_count", std::to_string(route_tree_by_net.size())}});

  std::string no_op_reason = "no_optimizable_clock";
  for (std::size_t clock_index = 0U; clock_index < clocks.size(); ++clock_index) {
    auto* clock = clocks.at(clock_index);
    if (clock == nullptr) {
      continue;
    }
    const auto clock_start = std::chrono::steady_clock::now();
    oi::ClockSizingRuntimeProfile outer_profile;
    outer_profile.build_route_tree_cache_s = route_tree_cache_runtime_s;
    stage_start = std::chrono::steady_clock::now();
    const auto route_geometry = oi::BuildClockRouteGeometry(clock_layout, clock_index);
    FastStaClockContextGuard clock_context(fast_sta, fast_sta.buildClockContext(FastStaClockBuildInput{
                                                         .clock = clock,
                                                         .route_geometry = &route_geometry,
                                                     }));
    const auto clock_id = clock_context.id();
    outer_profile.build_fast_sta_context_s = oi::ElapsedSeconds(stage_start);

    auto graph_profile = oi::CaptureGraphProfile(fast_sta, clock_id);
    graph_profile.build_route_tree_cache_s = outer_profile.build_route_tree_cache_s;
    graph_profile.build_fast_sta_context_s = outer_profile.build_fast_sta_context_s;
    outer_profile = graph_profile;

    stage_start = std::chrono::steady_clock::now();
    if (!oi::InjectRouteTrees(design, fast_sta, clock_id, *clock, route_tree_by_net)) {
      outer_profile.inject_route_trees_s = oi::ElapsedSeconds(stage_start);
      oi::EmitClockProfile(oi::ClockSizingProfileReportInput{
          .reporter = &reporter,
          .clock = clock,
          .profile = &outer_profile,
      });
      LOG_WARNING << "Optimization: skip clock \"" << clock->get_clock_name() << "\" because fast STA context build failed.";
      no_op_reason = "fast_sta_context_failed";
      continue;
    }
    outer_profile.inject_route_trees_s = oi::ElapsedSeconds(stage_start);

    stage_start = std::chrono::steady_clock::now();
    auto buffers = oi::CollectClockSizingBuffers(design, fast_sta, clock_id, master_infos);
    outer_profile.collect_optimizable_buffers_s = oi::ElapsedSeconds(stage_start);
    outer_profile.optimizable_buffer_count = buffers.size();
    if (buffers.empty()) {
      oi::EmitClockProfile(oi::ClockSizingProfileReportInput{
          .reporter = &reporter,
          .clock = clock,
          .profile = &outer_profile,
      });
      LOG_WARNING << "Optimization: skip clock \"" << clock->get_clock_name() << "\" because no resizable buffers are available.";
      no_op_reason = "no_resizable_buffers";
      continue;
    }

    stage_start = std::chrono::steady_clock::now();
    const auto cap_baseline = oi::CollectClockSizingCapLimits(fast_sta, clock_id);
    outer_profile.collect_cap_baseline_s = oi::ElapsedSeconds(stage_start);
    stage_start = std::chrono::steady_clock::now();
    const auto slew_baseline = oi::CollectClockSizingSlewLimits(fast_sta, clock_id);
    outer_profile.collect_slew_baseline_s = oi::ElapsedSeconds(stage_start);
    stage_start = std::chrono::steady_clock::now();
    const bool use_scalable_solver = oi::ShouldUseScalableSolver(oi::ScalableSolverDecisionInput{
        .fast_sta = &fast_sta,
        .clock_id = clock_id,
        .buffers = &buffers,
    });
    LOG_INFO << "Optimization: clock \"" << clock->get_clock_name() << "\" uses "
             << (use_scalable_solver ? "scalable timing-only batch solver" : "exact full-power batch solver") << ".";
    auto summary = use_scalable_solver ? oi::SolveClockScalable(fast_sta, clock_id, buffers, cap_baseline, slew_baseline, target_skew_ns)
                                       : oi::SolveClock(fast_sta, clock_id, buffers, cap_baseline, slew_baseline, target_skew_ns);
    outer_profile.solve_clock_s = oi::ElapsedSeconds(stage_start);
    oi::CopyOuterProfile(summary.profile, outer_profile);
    const auto clock_end = std::chrono::steady_clock::now();
    const double clock_runtime_s = std::chrono::duration<double>(clock_end - clock_start).count();
    if (!summary.valid) {
      oi::EmitClockProfile(oi::ClockSizingProfileReportInput{
          .reporter = &reporter,
          .clock = clock,
          .profile = &summary.profile,
      });
      LOG_WARNING << "Optimization: skip clock \"" << clock->get_clock_name() << "\" because fast STA solver failed with reason "
                  << summary.stop_reason << ".";
      no_op_reason = summary.stop_reason.empty() ? "solver_failed" : summary.stop_reason;
      continue;
    }
    stage_start = std::chrono::steady_clock::now();
    if (!summary.accepted_edits.empty()
        && !oi::ApplyClockSizingAcceptedEdits(design, wrapper, summary.accepted_edits, buffers, clock_layout)) {
      optimization_summary.success = false;
      optimization_summary.status = "failed";
      optimization_summary.reason = "accepted_edit_apply_failed";
      (void) runtime.failed();
      stage.failed({{"reason", "accepted_edit_apply_failed"}});
      return optimization_summary;
    }
    summary.profile.apply_accepted_edits_s = oi::ElapsedSeconds(stage_start);
    oi::EmitClockSummary(reporter, *clock, summary, target_skew_ns, clock_runtime_s);
    oi::EmitClockProfile(oi::ClockSizingProfileReportInput{
        .reporter = &reporter,
        .clock = clock,
        .profile = &summary.profile,
    });
    if (summary.accepted_edits.empty() && !summary.stop_reason.empty()
        && (no_op_reason == "no_optimizable_clock" || no_op_reason == "target_met")) {
      no_op_reason = summary.stop_reason;
    }
    optimization_summary.optimized = optimization_summary.optimized || !summary.accepted_edits.empty();
    optimization_summary.optimized_clock_count += summary.accepted_edits.empty() ? 0U : 1U;
    optimization_summary.accepted_edit_count += summary.accepted_edit_count;
  }

  const auto end_time = std::chrono::steady_clock::now();
  const double total_runtime_s = std::chrono::duration<double>(end_time - start_time).count();
  EmitKeyValueTable(reporter, "CTS Optimization Summary",
                    {
                        {"runtime", logformat::FormatWithUnit(total_runtime_s, "s")},
                        {"clock_count", std::to_string(optimization_summary.clock_count)},
                        {"optimized_clock_count", std::to_string(optimization_summary.optimized_clock_count)},
                        {"accepted_edit_count", std::to_string(optimization_summary.accepted_edit_count)},
                        {"status", optimization_summary.optimized ? "optimized" : "no_op"},
                    });
  LOG_INFO << "CTS optimization finished with " << optimization_summary.accepted_edit_count << " accepted sizing edits across "
           << optimization_summary.optimized_clock_count << " clocks.";
  if (optimization_summary.optimized) {
    optimization_summary.status = "optimized";
    (void) runtime.finished();
    stage.finished({{"accepted_edit_count", std::to_string(optimization_summary.accepted_edit_count)}});
  } else {
    optimization_summary.status = "no_op";
    optimization_summary.reason = no_op_reason;
    (void) runtime.finish("no_op");
    stage.skip({{"reason", no_op_reason}});
  }
  return optimization_summary;
}

}  // namespace icts
