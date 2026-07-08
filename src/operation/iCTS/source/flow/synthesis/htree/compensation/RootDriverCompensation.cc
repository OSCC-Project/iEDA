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
 * @file RootDriverCompensation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-04
 * @brief H-tree root driver compensation pass implementation.
 */

#include "synthesis/htree/compensation/RootDriverCompensation.hh"

#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <ratio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "HTreeTopologyChar.hh"
#include "HTreeTopologyPattern.hh"
#include "Log.hh"
#include "PatternId.hh"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/compensation/RootDriverCompensationState.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"

namespace icts::htree {
namespace {

auto HashCombine(std::size_t seed, std::size_t value) -> std::size_t
{
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

auto CoveringLatticeIndex(double value, const UniformValueLattice& lattice) -> unsigned
{
  if (value <= 0.0 || !lattice.isValid()) {
    return 0U;
  }
  return lattice.coveringIndex(value);
}

auto ResolveRootDriverCellMaster(PatternId topology_pattern_id, const TopologyPatternLibrary& topology_library,
                                 const BufferPatternLibrary& segment_pattern_library, const std::string& default_cell_master) -> std::string
{
  const auto topology_pattern = topology_library.materialize(topology_pattern_id);
  for (const auto segment_pattern_id : topology_pattern.get_level_segment_pattern_ids()) {
    const auto* segment_pattern = segment_pattern_library.find(segment_pattern_id);
    LOG_FATAL_IF(segment_pattern == nullptr) << "HTree: candidate segment pattern metadata is missing.";
    const auto& cell_masters = segment_pattern->get_cell_masters();
    if (!cell_masters.empty()) {
      return cell_masters.back();
    }
  }
  return default_cell_master;
}

auto MakeRootDriverCompensationDetail(const Wrapper::RootDriverCost& cost, double input_slew_ns,
                                      const RootClosureLoadEstimate& load_estimate, double clock_period_ns,
                                      const UniformValueLattice& slew_lattice) -> RootDriverCompensationDetail
{
  RootDriverCompensationDetail detail;
  detail.enabled = true;
  detail.valid = cost.valid;
  detail.method = kRootDriverCompensationMethod;
  detail.cell_master = cost.cell_master;
  detail.load_source = load_estimate.source;
  detail.route_estimator = load_estimate.route_estimator;
  detail.input_slew_ns = input_slew_ns;
  detail.load_bucket_idx = load_estimate.bucket_idx;
  detail.load_cap_pf = load_estimate.total_load_cap_pf;
  detail.source_boundary_bucket_idx = load_estimate.source_boundary_bucket_idx;
  detail.source_boundary_load_cap_pf = load_estimate.source_boundary_load_cap_pf;
  detail.source_boundary_branch_count = load_estimate.source_boundary_branch_count;
  detail.terminal_pin_cap_pf = load_estimate.terminal_pin_cap_pf;
  detail.wire_cap_pf = load_estimate.wire_cap_pf;
  detail.routed_wirelength_um = load_estimate.routed_wirelength_um;
  detail.terminal_count = load_estimate.terminal_count;
  detail.clock_period_ns = clock_period_ns;
  detail.output_slew_ns = cost.output_slew_ns;
  detail.output_slew_bucket_idx = CoveringLatticeIndex(cost.output_slew_ns, slew_lattice);
  detail.cell_delay_ns = cost.cell_delay_ns;
  detail.internal_power_w = cost.internal_power_w;
  detail.leakage_power_w = cost.leakage_power_w;
  detail.cell_power_w = cost.cell_power_w;
  return detail;
}

auto QueryRootDriverCompensation(const RootDriverCompensationCacheKey& key, const RootClosureLoadEstimate& load_estimate,
                                 RootDriverCompensationState& state) -> RootDriverCompensationDetail
{
  auto& stats = state.stats;
  const auto cache_it = state.cost_by_key.find(key);
  if (cache_it != state.cost_by_key.end()) {
    ++stats.cache_hit_count;
    auto result = cache_it->second;
    result.load_source = load_estimate.source;
    result.route_estimator = load_estimate.route_estimator;
    result.load_bucket_idx = load_estimate.bucket_idx;
    result.load_cap_pf = load_estimate.total_load_cap_pf;
    result.source_boundary_bucket_idx = load_estimate.source_boundary_bucket_idx;
    result.source_boundary_load_cap_pf = load_estimate.source_boundary_load_cap_pf;
    result.source_boundary_branch_count = load_estimate.source_boundary_branch_count;
    result.terminal_pin_cap_pf = load_estimate.terminal_pin_cap_pf;
    result.wire_cap_pf = load_estimate.wire_cap_pf;
    result.routed_wirelength_um = load_estimate.routed_wirelength_um;
    result.terminal_count = load_estimate.terminal_count;
    return result;
  }

  const auto lookup_start = std::chrono::steady_clock::now();
  LOG_FATAL_IF(state.input.wrapper == nullptr) << "HTree: Wrapper is unavailable for root-driver compensation.";
  const auto cost
      = state.input.wrapper->queryRootDriverCostDirect(key.cell_master, key.input_slew_ns, key.load_cap_pf, key.clock_period_ns);
  stats.total_runtime_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lookup_start).count();
  ++stats.unique_direct_lookup_count;
  auto compensation
      = MakeRootDriverCompensationDetail(cost, key.input_slew_ns, load_estimate, key.clock_period_ns, state.input.slew_lattice);
  return state.cost_by_key.emplace(key, std::move(compensation)).first->second;
}

auto RefreshCompensationStats(RootDriverCompensationState& state) -> void
{
  auto& stats = state.stats;
  stats.enabled = state.input.enabled;
  stats.method = state.input.enabled ? kRootDriverCompensationMethod : "disabled";
  stats.input_slew_ns = state.input.input_slew_ns;
  stats.clock_period_ns = state.input.clock_period_ns;
  stats.load_source = state.input.enabled ? kRootDriverCompensationLoadSource : "none";
}

auto CompensationInputIsValid(RootDriverCompensationState& state) -> bool
{
  if (state.input.input_slew_ns >= 0.0 && state.input.clock_period_ns > 0.0 && state.input.cap_lattice.isValid()
      && (!state.input.strict_boundary_closure || state.input.slew_lattice.isValid())) {
    return true;
  }

  if (!state.warned_invalid_input) {
    LOG_WARNING << "HTree: root-driver direct compensation skipped because input slew, clock period, or cap/slew lattice input "
                   "are invalid.";
    state.warned_invalid_input = true;
  }
  return false;
}

auto EvaluateRootDriverCompensation(PatternId pattern_id, const TopologyPatternLibrary& topology_library,
                                    const BufferPatternLibrary& segment_pattern_library, const Tree& topology,
                                    RootDriverCompensationState& compensation_state) -> RootDriverCompensationDetail
{
  if (!compensation_state.input.enabled || !CompensationInputIsValid(compensation_state)) {
    return {};
  }

  const auto cell_master
      = ResolveRootDriverCellMaster(pattern_id, topology_library, segment_pattern_library, compensation_state.input.default_cell_master);
  if (cell_master.empty()) {
    return {};
  }

  const auto load_estimate
      = QueryRootClosureLoadEstimate(pattern_id, topology_library, segment_pattern_library, topology, compensation_state);
  if (!load_estimate.valid) {
    return {};
  }

  const RootDriverCompensationCacheKey key{
      .cell_master = cell_master,
      .input_slew_ns = compensation_state.input.input_slew_ns,
      .load_bucket_idx = load_estimate.bucket_idx,
      .load_cap_pf = load_estimate.total_load_cap_pf,
      .clock_period_ns = compensation_state.input.clock_period_ns,
  };
  return QueryRootDriverCompensation(key, load_estimate, compensation_state);
}

auto CheckRootDriverBoundaryClosure(const HTreeTopologyChar& entry, const TopologyPatternLibrary& topology_library,
                                    const BufferPatternLibrary& segment_pattern_library, const Tree& topology,
                                    RootDriverCompensationState& compensation_state) -> RootDriverBoundaryClosureCheck
{
  auto compensation
      = EvaluateRootDriverCompensation(entry.get_pattern_id(), topology_library, segment_pattern_library, topology, compensation_state);
  RootDriverBoundaryClosureCheck check;
  check.compensation_valid = compensation.enabled && compensation.valid && compensation.load_bucket_idx > 0U
                             && compensation.source_boundary_bucket_idx > 0U && compensation.output_slew_bucket_idx > 0U;
  check.raw_cap_bucket_idx = entry.get_driven_cap_idx();
  check.physical_load_bucket_idx = compensation.load_bucket_idx;
  check.physical_source_boundary_bucket_idx = compensation.source_boundary_bucket_idx;
  check.raw_input_slew_idx = entry.get_input_slew_idx();
  check.root_output_slew_bucket_idx = compensation.output_slew_bucket_idx;
  check.cap_bucket_matches = check.compensation_valid && check.raw_cap_bucket_idx >= check.physical_source_boundary_bucket_idx;
  check.slew_bucket_matches = check.compensation_valid && check.raw_input_slew_idx >= check.root_output_slew_bucket_idx;
  check.compensation = std::move(compensation);
  return check;
}

}  // namespace

auto ResolveRootDriverCompensationInputSlewNs(const HTree::Config& config, double max_slew_ns) -> double
{
  if (config.min_top_input_slew_ns.has_value() && *config.min_top_input_slew_ns >= 0.0) {
    return *config.min_top_input_slew_ns;
  }
  return max_slew_ns > 0.0 ? max_slew_ns * 0.5 : 0.0;
}

auto ResolveRootDriverClockPeriod(const HTree::Input& input) -> std::pair<double, std::string>
{
  if (input.clock_period_ns > 0.0) {
    return {input.clock_period_ns, input.clock_period_source.empty() ? "caller" : input.clock_period_source};
  }
  return {kRootDriverCompensationClockPeriodNs, "default_10ns"};
}

auto ApplyRootDriverCompensationSummary(htree::DiagnosticBuild& build, const RootDriverCompensationStats& compensation_stats,
                                        const RootDriverCompensationDetail& compensation_detail, const HTreeTopologyChar& selected_entry)
    -> void
{
  auto& report = build.diagnostics.root_driver_compensation;
  report.enabled = compensation_stats.enabled;
  report.valid = compensation_detail.valid;
  report.method = compensation_detail.method.empty() ? compensation_stats.method : compensation_detail.method;
  report.cell_master = compensation_detail.cell_master;
  report.load_source = compensation_detail.load_source.empty() ? compensation_stats.load_source : compensation_detail.load_source;
  report.route_estimator = compensation_detail.route_estimator;
  report.input_slew_ns = compensation_detail.input_slew_ns > 0.0 ? compensation_detail.input_slew_ns : compensation_stats.input_slew_ns;
  report.load_bucket_idx = compensation_detail.load_bucket_idx;
  report.load_cap_pf = compensation_detail.load_cap_pf;
  report.source_boundary_bucket_idx = compensation_detail.source_boundary_bucket_idx;
  report.source_boundary_load_cap_pf = compensation_detail.source_boundary_load_cap_pf;
  report.source_boundary_branch_count = compensation_detail.source_boundary_branch_count;
  report.terminal_pin_cap_pf = compensation_detail.terminal_pin_cap_pf;
  report.wire_cap_pf = compensation_detail.wire_cap_pf;
  report.routed_wirelength_um = compensation_detail.routed_wirelength_um;
  report.terminal_count = compensation_detail.terminal_count;
  report.clock_period_ns
      = compensation_detail.clock_period_ns > 0.0 ? compensation_detail.clock_period_ns : compensation_stats.clock_period_ns;
  report.output_slew_ns = compensation_detail.output_slew_ns;
  report.output_slew_bucket_idx = compensation_detail.output_slew_bucket_idx;
  report.cell_delay_ns = compensation_detail.cell_delay_ns;
  report.internal_power_w = compensation_detail.internal_power_w;
  report.leakage_power_w = compensation_detail.leakage_power_w;
  report.cell_power_w = compensation_detail.cell_power_w;
  report.raw_delay_ns = selected_entry.get_raw_delay();
  report.raw_power_w = selected_entry.get_raw_power();
  report.compensated_delay_ns = selected_entry.get_delay();
  report.compensated_power_w = selected_entry.get_power();
}

auto ApplyRootDriverCompensationSummary(htree::DiagnosticBuild& build, const DepthSearchBuild& exploration,
                                        const RootDriverCompensationDetail& compensation_detail, const HTreeTopologyChar& selected_entry)
    -> void
{
  ApplyRootDriverCompensationSummary(build, exploration.summary.root_driver_compensation_stats, compensation_detail, selected_entry);
}

auto RootDriverCompensationCacheKeyHash::operator()(const RootDriverCompensationCacheKey& key) const noexcept -> std::size_t
{
  std::size_t seed = std::hash<std::string>{}(key.cell_master);
  seed = HashCombine(seed, std::hash<double>{}(key.input_slew_ns));
  seed = HashCombine(seed, std::hash<unsigned>{}(key.load_bucket_idx));
  seed = HashCombine(seed, std::hash<double>{}(key.load_cap_pf));
  seed = HashCombine(seed, std::hash<double>{}(key.clock_period_ns));
  return seed;
}

struct RootDriverCompensationPass::Impl
{
  explicit Impl(RootDriverCompensationInput input)
  {
    state.input = std::move(input);
    state.stats.enabled = state.input.enabled;
    state.stats.method = state.input.enabled ? kRootDriverCompensationMethod : "disabled";
    state.stats.input_slew_ns = state.input.input_slew_ns;
    state.stats.clock_period_ns = state.input.clock_period_ns;
    state.stats.load_source = state.input.enabled ? kRootDriverCompensationLoadSource : "none";
  }

  RootDriverCompensationState state;
};

RootDriverCompensationPass::RootDriverCompensationPass(RootDriverCompensationInput input) : _impl(std::make_unique<Impl>(std::move(input)))
{
}

RootDriverCompensationPass::~RootDriverCompensationPass() = default;

RootDriverCompensationPass::RootDriverCompensationPass(RootDriverCompensationPass&&) noexcept = default;

auto RootDriverCompensationPass::operator=(RootDriverCompensationPass&&) noexcept -> RootDriverCompensationPass& = default;

auto RootDriverCompensationPass::beginCandidateBuild() -> void
{
  _impl->state.warned_invalid_input = false;
}

auto RootDriverCompensationPass::apply(std::vector<HTreeTopologyChar>& entries, const TopologyPatternLibrary& topology_library,
                                       const BufferPatternLibrary& segment_pattern_library, const Tree& topology)
    -> RootDriverCompensationApplySummary
{
  RootDriverCompensationApplySummary apply_result;
  auto& compensation_state = _impl->state;
  RefreshCompensationStats(compensation_state);
  if (!compensation_state.input.enabled || entries.empty()) {
    return apply_result;
  }
  LOG_FATAL_IF(compensation_state.input.reporter == nullptr) << "HTree root-driver compensation requires an explicit reporter.";
  auto& reporter = *compensation_state.input.reporter;
  auto compensation_stage
      = reporter.beginStage("HTreeDepth", "Apply root-driver compensation",
                            {
                                {"entries", std::to_string(entries.size())},
                                {"input_slew_ns", std::to_string(compensation_state.input.input_slew_ns)},
                                {"clock_period_ns", std::to_string(compensation_state.input.clock_period_ns)},
                            },
                            StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail});
  if (!CompensationInputIsValid(compensation_state)) {
    compensation_stage.skip({{"reason", "invalid_compensation_options"}});
    return apply_result;
  }

  const auto unique_lookup_count_before = compensation_state.stats.unique_direct_lookup_count;
  const auto cache_hit_count_before = compensation_state.stats.cache_hit_count;
  const auto load_resolution_count_before = compensation_state.stats.load_resolution_count;
  const auto load_resolution_cache_hit_count_before = compensation_state.stats.load_resolution_cache_hit_count;
  const auto flute_route_estimate_count_before = compensation_state.stats.flute_route_estimate_count;
  const auto hpwl_route_estimate_count_before = compensation_state.stats.hpwl_route_estimate_count;
  const auto compensated_candidate_count_before = compensation_state.stats.compensated_candidate_count;
  const auto boundary_input_candidate_count_before = compensation_state.stats.boundary_input_candidate_count;
  const auto boundary_closed_candidate_count_before = compensation_state.stats.boundary_closed_candidate_count;
  const auto boundary_rejected_candidate_count_before = compensation_state.stats.boundary_rejected_candidate_count;
  const auto boundary_cap_bucket_mismatch_count_before = compensation_state.stats.boundary_cap_bucket_mismatch_count;
  const auto boundary_slew_bucket_mismatch_count_before = compensation_state.stats.boundary_slew_bucket_mismatch_count;
  const auto invalid_compensation_count_before = compensation_state.stats.invalid_compensation_count;
  std::vector<HTreeTopologyChar> boundary_closed_entries;
  if (compensation_state.input.strict_boundary_closure) {
    boundary_closed_entries.reserve(entries.size());
  }
  for (auto& entry : entries) {
    ++compensation_state.stats.boundary_input_candidate_count;
    ++apply_result.input_candidate_count;
    auto boundary_check = CheckRootDriverBoundaryClosure(entry, topology_library, segment_pattern_library, topology, compensation_state);
    if (!boundary_check.compensation_valid) {
      ++compensation_state.stats.invalid_compensation_count;
      if (compensation_state.input.strict_boundary_closure) {
        ++compensation_state.stats.boundary_rejected_candidate_count;
        ++apply_result.rejected_candidate_count;
        if (!apply_result.has_first_rejected_boundary) {
          apply_result.first_rejected_boundary = boundary_check;
          apply_result.has_first_rejected_boundary = true;
        }
        continue;
      }
    } else if (compensation_state.input.strict_boundary_closure) {
      if (!boundary_check.cap_bucket_matches) {
        ++compensation_state.stats.boundary_cap_bucket_mismatch_count;
      }
      if (!boundary_check.slew_bucket_matches) {
        ++compensation_state.stats.boundary_slew_bucket_mismatch_count;
      }
      if (!boundary_check.isClosed(compensation_state.input.strict_slew_boundary_closure)) {
        ++compensation_state.stats.boundary_rejected_candidate_count;
        ++apply_result.rejected_candidate_count;
        if (!apply_result.has_first_rejected_boundary) {
          apply_result.first_rejected_boundary = boundary_check;
          apply_result.has_first_rejected_boundary = true;
        }
        continue;
      }
      ++compensation_state.stats.boundary_closed_candidate_count;
      ++apply_result.closed_candidate_count;
    }
    if (!boundary_check.compensation.enabled) {
      continue;
    }
    entry.set_root_driver_compensation(boundary_check.compensation.cell_delay_ns, boundary_check.compensation.cell_power_w);
    ++compensation_state.stats.compensated_candidate_count;
    if (compensation_state.input.strict_boundary_closure) {
      boundary_closed_entries.push_back(std::move(entry));
    }
  }
  if (compensation_state.input.strict_boundary_closure) {
    entries = std::move(boundary_closed_entries);
  }
  compensation_stage.finished({
      {"compensated_candidates", std::to_string(compensation_state.stats.compensated_candidate_count - compensated_candidate_count_before)},
      {"strict_boundary_closure", compensation_state.input.strict_boundary_closure ? "true" : "false"},
      {"strict_slew_boundary_closure", compensation_state.input.strict_slew_boundary_closure ? "true" : "false"},
      {"boundary_input_candidates",
       std::to_string(compensation_state.stats.boundary_input_candidate_count - boundary_input_candidate_count_before)},
      {"boundary_closed_candidates",
       std::to_string(compensation_state.stats.boundary_closed_candidate_count - boundary_closed_candidate_count_before)},
      {"boundary_rejected_candidates",
       std::to_string(compensation_state.stats.boundary_rejected_candidate_count - boundary_rejected_candidate_count_before)},
      {"cap_bucket_mismatches",
       std::to_string(compensation_state.stats.boundary_cap_bucket_mismatch_count - boundary_cap_bucket_mismatch_count_before)},
      {"slew_bucket_mismatches",
       std::to_string(compensation_state.stats.boundary_slew_bucket_mismatch_count - boundary_slew_bucket_mismatch_count_before)},
      {"invalid_compensations", std::to_string(compensation_state.stats.invalid_compensation_count - invalid_compensation_count_before)},
      {"unique_direct_lookups", std::to_string(compensation_state.stats.unique_direct_lookup_count - unique_lookup_count_before)},
      {"direct_cache_hits", std::to_string(compensation_state.stats.cache_hit_count - cache_hit_count_before)},
      {"load_resolutions", std::to_string(compensation_state.stats.load_resolution_count - load_resolution_count_before)},
      {"load_resolution_cache_hits",
       std::to_string(compensation_state.stats.load_resolution_cache_hit_count - load_resolution_cache_hit_count_before)},
      {"flute_route_estimates", std::to_string(compensation_state.stats.flute_route_estimate_count - flute_route_estimate_count_before)},
      {"hpwl_route_estimates", std::to_string(compensation_state.stats.hpwl_route_estimate_count - hpwl_route_estimate_count_before)},
  });
  return apply_result;
}

auto RootDriverCompensationPass::evaluate(PatternId pattern_id, const TopologyPatternLibrary& topology_library,
                                          const BufferPatternLibrary& segment_pattern_library, const Tree& topology)
    -> RootDriverCompensationDetail
{
  auto& compensation_state = _impl->state;
  RefreshCompensationStats(compensation_state);
  return EvaluateRootDriverCompensation(pattern_id, topology_library, segment_pattern_library, topology, compensation_state);
}

auto RootDriverCompensationPass::get_stats() const -> const RootDriverCompensationStats&
{
  return _impl->state.stats;
}

}  // namespace icts::htree
