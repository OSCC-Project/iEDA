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
 * @file TopologyPruning.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree topology frontier composition and candidate selection.
 */

#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "CharCore.hh"
#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "PatternId.hh"
#include "SegmentChar.hh"
#include "characterization/Characterization.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/segment_pruning/SegmentFrontierCatalog.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"

namespace icts::htree {
namespace {

auto CompactPatternSearchToFrontier(PatternSearchBuild& result, std::vector<HTreeTopologyChar>& frontier_entries) -> void;

auto BuildDelayPowerParetoFront(const std::vector<HTreeTopologyChar>& entries) -> std::vector<const HTreeTopologyChar*>
{
  std::vector<const HTreeTopologyChar*> pareto_front;
  pareto_front.reserve(entries.size());

  std::vector<const HTreeTopologyChar*> sorted_entries;
  sorted_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    sorted_entries.push_back(&entry);
  }

  std::ranges::sort(sorted_entries, [](const HTreeTopologyChar* lhs, const HTreeTopologyChar* rhs) -> bool {
    LOG_FATAL_IF(lhs == nullptr || rhs == nullptr) << "HTree: null pareto entry encountered during delay-power sorting.";
    if (lhs->get_delay() != rhs->get_delay()) {
      return lhs->get_delay() < rhs->get_delay();
    }
    if (lhs->get_power() != rhs->get_power()) {
      return lhs->get_power() < rhs->get_power();
    }
    return lhs->get_pattern_id().pack() < rhs->get_pattern_id().pack();
  });

  double best_power_before_delay = std::numeric_limits<double>::infinity();
  std::size_t delay_group_begin = 0U;
  while (delay_group_begin < sorted_entries.size()) {
    std::size_t delay_group_end = delay_group_begin + 1U;
    while (delay_group_end < sorted_entries.size()
           && sorted_entries.at(delay_group_end)->get_delay() == sorted_entries.at(delay_group_begin)->get_delay()) {
      ++delay_group_end;
    }

    double delay_group_min_power = std::numeric_limits<double>::infinity();
    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      delay_group_min_power = std::min(delay_group_min_power, sorted_entries.at(index)->get_power());
    }

    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      const auto* entry = sorted_entries.at(index);
      if (best_power_before_delay <= entry->get_power() || delay_group_min_power < entry->get_power()) {
        continue;
      }
      pareto_front.push_back(entry);
    }

    best_power_before_delay = std::min(best_power_before_delay, delay_group_min_power);
    delay_group_begin = delay_group_end;
  }
  return pareto_front;
}

auto PreferDelayPowerTieBreak(const HTreeTopologyChar& lhs, const HTreeTopologyChar& rhs) -> bool
{
  return (lhs.get_driven_cap_idx() < rhs.get_driven_cap_idx())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() < rhs.get_output_slew_idx())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() == rhs.get_output_slew_idx()
             && lhs.get_delay() < rhs.get_delay())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() == rhs.get_output_slew_idx()
             && lhs.get_delay() == rhs.get_delay() && lhs.get_power() < rhs.get_power())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() == rhs.get_output_slew_idx()
             && lhs.get_delay() == rhs.get_delay() && lhs.get_power() == rhs.get_power() && lhs.get_load_cap_idx() < rhs.get_load_cap_idx())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() == rhs.get_output_slew_idx()
             && lhs.get_delay() == rhs.get_delay() && lhs.get_power() == rhs.get_power() && lhs.get_load_cap_idx() == rhs.get_load_cap_idx()
             && lhs.get_input_slew_idx() > rhs.get_input_slew_idx())
         || (lhs.get_driven_cap_idx() == rhs.get_driven_cap_idx() && lhs.get_output_slew_idx() == rhs.get_output_slew_idx()
             && lhs.get_delay() == rhs.get_delay() && lhs.get_power() == rhs.get_power() && lhs.get_load_cap_idx() == rhs.get_load_cap_idx()
             && lhs.get_input_slew_idx() == rhs.get_input_slew_idx() && lhs.get_pattern_id().pack() < rhs.get_pattern_id().pack());
}

auto PreferPowerMedianOrder(const HTreeTopologyChar& lhs, const HTreeTopologyChar& rhs) -> bool
{
  if (lhs.get_power() != rhs.get_power()) {
    return lhs.get_power() < rhs.get_power();
  }
  if (lhs.get_delay() != rhs.get_delay()) {
    return lhs.get_delay() < rhs.get_delay();
  }
  return PreferDelayPowerTieBreak(lhs, rhs);
}

auto ResolveSegmentFrontierKind(const BoundaryConstraints& boundary_constraints) -> SegmentFrontierKind
{
  if (boundary_constraints.force_branch_buffer) {
    return SegmentFrontierKind::kTerminalBranchBuffered;
  }
  return SegmentFrontierKind::kAll;
}

auto SelectSegmentEntriesForLevel(const SegmentFrontierCatalog& segment_frontier_catalog, unsigned length_idx,
                                  SegmentFrontierKind frontier_kind) -> const std::vector<SegmentChar>*
{
  return segment_frontier_catalog.find(length_idx, frontier_kind);
}

auto DescribeMissingSegmentEntries(SegmentFrontierKind frontier_kind) -> std::string
{
  switch (frontier_kind) {
    case SegmentFrontierKind::kTerminalBranchBuffered:
      return "missing_branch_buffered_segment_frontier";
    case SegmentFrontierKind::kTerminalLeafUnbuffered:
      return "missing_leaf_unbuffered_segment_frontier";
    case SegmentFrontierKind::kAll:
      return "missing_segment_frontier";
  }
  return "missing_segment_frontier";
}

auto MakeHTreeSeedEntries(const std::vector<SegmentChar>& segment_frontier, const BufferPatternLibrary& segment_pattern_library,
                          TopologyPatternLibrary& topology_library, unsigned& next_pattern_id) -> std::vector<HTreeTopologyChar>
{
  std::vector<HTreeTopologyChar> seed_entries;
  seed_entries.reserve(segment_frontier.size());
  for (const auto& segment_entry : segment_frontier) {
    const auto topology_pattern_id = PatternId::topology(next_pattern_id++);
    topology_library.addSeed(topology_pattern_id, segment_entry.get_pattern_id(),
                             segment_pattern_library.getCompositionState(segment_entry.get_pattern_id()));
    const CharCore core(segment_entry.get_input_slew_idx(), segment_entry.get_output_slew_idx(), segment_entry.get_driven_cap_idx(),
                        segment_entry.get_load_cap_idx(), segment_entry.get_delay(), segment_entry.get_power(), topology_pattern_id,
                        segment_entry.get_source_boundary_net_switch_power());
    seed_entries.emplace_back(core, 1U);
  }
  return seed_entries;
}

auto IsRootExposedFanoutLegal(const HTreeTopologyChar& entry, const TopologyPatternLibrary& topology_library,
                              const HTreeFanoutPruningConfig& fanout_config) -> bool
{
  const auto source_load_count = topology_library.getCompositionState(entry.get_pattern_id()).source_exposed_load_count;
  return IsBinarySourceFanoutLegal(source_load_count, fanout_config.max_fanout);
}

auto FilterRootFanoutLegalHTreeChars(const std::vector<HTreeTopologyChar>& entries, const TopologyPatternLibrary& topology_library,
                                     const HTreeFanoutPruningConfig& fanout_config) -> std::vector<HTreeTopologyChar>
{
  if (fanout_config.max_fanout == 0U) {
    return entries;
  }

  std::vector<HTreeTopologyChar> filtered_entries;
  filtered_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    if (IsRootExposedFanoutLegal(entry, topology_library, fanout_config)) {
      filtered_entries.push_back(entry);
    }
  }
  return filtered_entries;
}

auto BuildHTreeFrontierStateKey(const HTreeTopologyChar& entry, const PatternCompositionState& state) -> HTreeFrontierStateKey
{
  const auto normalized_state = NormalizePatternCompositionState(state);
  return HTreeFrontierStateKey{
      .input_slew_idx = entry.get_input_slew_idx(),
      .driven_cap_idx = entry.get_driven_cap_idx(),
      .leaf_load_cap_idx = entry.get_leaf_load_cap_idx(),
      .output_slew_idx = entry.get_output_slew_idx(),
      .load_cap_idx = entry.get_load_cap_idx(),
      .source_boundary_net_switch_power_w = entry.get_source_boundary_net_switch_power(),
      .terminal_semantic = normalized_state.terminal_semantic,
      .monotonic_boundary_state = normalized_state.monotonic_boundary_state,
      .source_exposed_load_count = normalized_state.source_exposed_load_count,
  };
}

auto ComposeHTreeFrontierEntries(const std::vector<HTreeTopologyChar>& upstream, const std::vector<HTreeTopologyChar>& downstream,
                                 TopologyPatternLibrary& topology_library, unsigned start_pattern_id,
                                 const HTreeFanoutPruningConfig& fanout_config) -> std::pair<std::vector<HTreeTopologyChar>, unsigned>
{
  if (upstream.empty() || downstream.empty()) {
    return {{}, start_pattern_id};
  }

  TopologyPatternLibraryCombiner combiner(topology_library, start_pattern_id, fanout_config.max_fanout);
  std::unordered_map<unsigned, std::vector<std::size_t>> downstream_entries_by_key;
  downstream_entries_by_key.reserve(downstream.size());
  for (std::size_t index = 0U; index < downstream.size(); ++index) {
    downstream_entries_by_key[HTreeTraits::buildKey(downstream.at(index))].push_back(index);
  }

  std::unordered_map<HTreeFrontierStateKey, std::vector<HTreeTopologyChar>, HTreeFrontierStateKeyHash> frontier_by_state;
  frontier_by_state.reserve(upstream.size());
  for (const auto& upstream_entry : upstream) {
    const auto downstream_it = downstream_entries_by_key.find(HTreeTraits::probeKey(upstream_entry));
    if (downstream_it == downstream_entries_by_key.end()) {
      continue;
    }

    for (const auto downstream_index : downstream_it->second) {
      const auto& downstream_entry = downstream.at(downstream_index);
      if (!combiner.canCompose(upstream_entry.get_pattern_id(), downstream_entry.get_pattern_id())) {
        continue;
      }

      const auto tentative_pattern_id = PatternId::topology(combiner.get_next_id());
      auto result = HTreeTopologyChar::compose(upstream_entry, downstream_entry, tentative_pattern_id);
      const auto composition_state = combiner.composeState(upstream_entry.get_pattern_id(), downstream_entry.get_pattern_id());
      auto& frontier = frontier_by_state[BuildHTreeFrontierStateKey(result, composition_state)];

      bool dominated = false;
      for (const auto& existing : frontier) {
        if (CostDominates(existing, result)) {
          dominated = true;
          break;
        }
      }
      if (dominated) {
        continue;
      }

      const auto removed_entries
          = std::ranges::remove_if(frontier, [&](const HTreeTopologyChar& existing) -> bool { return CostDominates(result, existing); });
      frontier.erase(removed_entries.begin(), removed_entries.end());

      const auto committed_pattern_id = combiner.combine(upstream_entry.get_pattern_id(), downstream_entry.get_pattern_id());
      LOG_FATAL_IF(committed_pattern_id != tentative_pattern_id) << "HTree: tentative topology pattern ID changed before commit.";
      frontier.push_back(std::move(result));
    }
  }

  std::size_t frontier_size = 0U;
  for (const auto& [state_key, entries] : frontier_by_state) {
    (void) state_key;
    frontier_size += entries.size();
  }
  std::vector<HTreeTopologyChar> frontier_entries;
  frontier_entries.reserve(frontier_size);
  for (auto& [state_key, entries] : frontier_by_state) {
    (void) state_key;
    frontier_entries.insert(frontier_entries.end(), std::make_move_iterator(entries.begin()), std::make_move_iterator(entries.end()));
  }
  SortHTreeFrontierEntries(frontier_entries);
  return {std::move(frontier_entries), combiner.get_next_id()};
}

auto CountBoundaryFeasibleHTreeChars(const std::vector<HTreeTopologyChar>& entries, const BoundaryConstraints& boundary_constraints)
    -> std::size_t
{
  if (!HasBoundaryConstraints(boundary_constraints) || !boundary_constraints.top_input_slew_covering_idx.has_value()) {
    return entries.size();
  }

  return static_cast<std::size_t>(std::ranges::count_if(entries, [&](const HTreeTopologyChar& entry) -> bool {
    return entry.get_input_slew_idx() >= *boundary_constraints.top_input_slew_covering_idx;
  }));
}

auto BuildPatternSearch(const std::vector<HTree::LevelPlan>& levels, const SegmentFrontierCatalog& segment_frontier_catalog,
                        const BufferPatternLibrary& segment_pattern_library, const BoundaryConstraints& boundary_constraints,
                        const Tree& topology, RootDriverCompensationPass& compensation_pass, SchemaWriter& reporter,
                        const HTreeFanoutPruningConfig& fanout_config) -> PatternSearchBuild
{
  auto pattern_search_stage
      = reporter.beginStage("HTreeDepth", "Build pattern frontier",
                            {
                                {"levels", std::to_string(levels.size())},
                                {"segment_frontier_length_sets", std::to_string(segment_frontier_catalog.lengthCount())},
                            },
                            StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail});
  PatternSearchBuild result;
  unsigned next_topology_pattern_id = 0U;
  std::vector<HTreeTopologyChar> current_frontier_entries;

  for (std::size_t reverse_level = levels.size(); reverse_level > 0U; --reverse_level) {
    const auto& level = levels.at(reverse_level - 1U);
    result.failure_level = static_cast<unsigned>(reverse_level - 1U);
    result.failure_length_idx = level.aligned_length_idx;

    if (segment_frontier_catalog.find(level.aligned_length_idx, SegmentFrontierKind::kAll) == nullptr) {
      result.failure_reason = "missing_segment_frontier";
      pattern_search_stage.failed({
          {"reason", result.failure_reason},
          {"failure_level", std::to_string(result.failure_level)},
          {"failure_length_idx", std::to_string(result.failure_length_idx)},
      });
      return result;
    }

    const SegmentFrontierKind frontier_kind = ResolveSegmentFrontierKind(boundary_constraints);
    const auto* base_segment_frontier = SelectSegmentEntriesForLevel(segment_frontier_catalog, level.aligned_length_idx, frontier_kind);
    if (base_segment_frontier == nullptr || base_segment_frontier->empty()) {
      result.failure_reason = DescribeMissingSegmentEntries(frontier_kind);
      pattern_search_stage.failed({
          {"reason", result.failure_reason},
          {"failure_level", std::to_string(result.failure_level)},
          {"failure_length_idx", std::to_string(result.failure_length_idx)},
      });
      return result;
    }

    auto seed_entries
        = MakeHTreeSeedEntries(*base_segment_frontier, segment_pattern_library, result.topology_pattern_library, next_topology_pattern_id);
    if (current_frontier_entries.empty()) {
      current_frontier_entries = std::move(seed_entries);
      continue;
    }

    auto [composed_entries, updated_next_pattern_id] = ComposeHTreeFrontierEntries(
        seed_entries, current_frontier_entries, result.topology_pattern_library, next_topology_pattern_id, fanout_config);
    next_topology_pattern_id = updated_next_pattern_id;
    current_frontier_entries = std::move(composed_entries);
    if (current_frontier_entries.empty()) {
      result.failure_reason = "empty_frontier";
      pattern_search_stage.failed({
          {"reason", result.failure_reason},
          {"failure_level", std::to_string(result.failure_level)},
          {"failure_length_idx", std::to_string(result.failure_length_idx)},
      });
      return result;
    }
    CompactPatternSearchToFrontier(result, current_frontier_entries);
    next_topology_pattern_id = static_cast<unsigned>(result.topology_pattern_library.nodes.size());
  }

  result.success = !current_frontier_entries.empty();
  const auto compensation_result
      = compensation_pass.apply(current_frontier_entries, result.topology_pattern_library, segment_pattern_library, topology);
  if (current_frontier_entries.empty()) {
    result.success = false;
    result.failure_reason = compensation_result.rejected_candidate_count > 0U ? "empty_frontier_after_root_boundary_closure"
                                                                              : "empty_frontier_after_compensation";
    pattern_search_stage.failed({
        {"reason", result.failure_reason},
        {"root_boundary_input_candidates", std::to_string(compensation_result.input_candidate_count)},
        {"root_boundary_closed_candidates", std::to_string(compensation_result.closed_candidate_count)},
        {"root_boundary_rejected_candidates", std::to_string(compensation_result.rejected_candidate_count)},
    });
    return result;
  }
  current_frontier_entries
      = BuildHTreeStateFrontier(current_frontier_entries, [&](const HTreeTopologyChar& entry) -> PatternCompositionState {
          return result.topology_pattern_library.getCompositionState(entry.get_pattern_id());
        });
  current_frontier_entries = FilterRootFanoutLegalHTreeChars(current_frontier_entries, result.topology_pattern_library, fanout_config);
  result.success = !current_frontier_entries.empty();
  result.frontier = std::move(current_frontier_entries);
  if (result.success) {
    pattern_search_stage.finished({
        {"frontier_entries", std::to_string(result.frontier.size())},
        {"topology_patterns", std::to_string(next_topology_pattern_id)},
    });
  } else {
    pattern_search_stage.failed({{"reason", "empty_frontier_after_compensation"}});
  }
  return result;
}

auto FilterBoundaryFeasibleHTreeChars(const std::vector<HTreeTopologyChar>& entries, const BoundaryConstraints& boundary_constraints)
    -> std::vector<HTreeTopologyChar>
{
  if (!HasBoundaryConstraints(boundary_constraints)) {
    return entries;
  }

  std::vector<HTreeTopologyChar> filtered_entries;
  filtered_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    if (boundary_constraints.top_input_slew_covering_idx.has_value()
        && entry.get_input_slew_idx() < *boundary_constraints.top_input_slew_covering_idx) {
      continue;
    }
    filtered_entries.push_back(entry);
  }
  return filtered_entries;
}

auto AppendTopologyPatternIds(const std::vector<HTreeTopologyChar>& entries, std::vector<PatternId>& pattern_ids) -> void
{
  pattern_ids.reserve(pattern_ids.size() + entries.size());
  for (const auto& entry : entries) {
    pattern_ids.push_back(entry.get_pattern_id());
  }
}

auto ReleaseTopologyPatternNodes(TopologyPatternLibrary& topology_pattern_library) -> void
{
  std::vector<TopologyPatternNode>().swap(topology_pattern_library.nodes);
}

auto RemapHTreeTopologyChars(const std::vector<HTreeTopologyChar>& entries, const std::unordered_map<PatternId, PatternId>& pattern_id_map)
    -> std::vector<HTreeTopologyChar>
{
  std::vector<HTreeTopologyChar> remapped_entries;
  remapped_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    const auto map_it = pattern_id_map.find(entry.get_pattern_id());
    LOG_FATAL_IF(map_it == pattern_id_map.end()) << "HTree: compacted topology library is missing a retained frontier pattern.";
    remapped_entries.push_back(entry.withPatternId(map_it->second));
  }
  return remapped_entries;
}

auto CompactCandidateEvaluationTopologyPatterns(CandidateBuildEvaluation& evaluation) -> void
{
  std::vector<PatternId> retained_pattern_ids;
  AppendTopologyPatternIds(evaluation.feasible_frontier_entries, retained_pattern_ids);
  AppendTopologyPatternIds(evaluation.candidate_frontier_entries, retained_pattern_ids);
  if (evaluation.best_char.has_value()) {
    retained_pattern_ids.push_back(evaluation.best_char->get_pattern_id());
  }
  if (retained_pattern_ids.empty()) {
    ReleaseTopologyPatternNodes(evaluation.topology_pattern_library);
    return;
  }

  auto [compact_library, pattern_id_map] = evaluation.topology_pattern_library.compactReachableFrom(retained_pattern_ids);
  auto remapped_feasible_entries = RemapHTreeTopologyChars(evaluation.feasible_frontier_entries, pattern_id_map);
  auto remapped_candidate_entries = RemapHTreeTopologyChars(evaluation.candidate_frontier_entries, pattern_id_map);
  if (evaluation.best_char.has_value()) {
    const auto map_it = pattern_id_map.find(evaluation.best_char->get_pattern_id());
    LOG_FATAL_IF(map_it == pattern_id_map.end()) << "HTree: compacted topology library is missing the selected candidate pattern.";
    evaluation.best_char = evaluation.best_char->withPatternId(map_it->second);
  }

  std::vector<HTreeTopologyChar>().swap(evaluation.feasible_frontier_entries);
  std::vector<HTreeTopologyChar>().swap(evaluation.candidate_frontier_entries);
  ReleaseTopologyPatternNodes(evaluation.topology_pattern_library);
  evaluation.feasible_frontier_entries = std::move(remapped_feasible_entries);
  evaluation.candidate_frontier_entries = std::move(remapped_candidate_entries);
  evaluation.topology_pattern_library = std::move(compact_library);
}

auto CompactPatternSearchToFrontier(PatternSearchBuild& result, std::vector<HTreeTopologyChar>& frontier_entries) -> void
{
  if (frontier_entries.empty()) {
    ReleaseTopologyPatternNodes(result.topology_pattern_library);
    return;
  }

  std::vector<PatternId> retained_pattern_ids;
  AppendTopologyPatternIds(frontier_entries, retained_pattern_ids);
  auto [compact_library, pattern_id_map] = result.topology_pattern_library.compactReachableFrom(retained_pattern_ids);
  auto remapped_frontier_entries = RemapHTreeTopologyChars(frontier_entries, pattern_id_map);
  std::vector<HTreeTopologyChar>().swap(frontier_entries);
  ReleaseTopologyPatternNodes(result.topology_pattern_library);
  frontier_entries = std::move(remapped_frontier_entries);
  result.topology_pattern_library = std::move(compact_library);
}

auto SelectBestHTreeChar(const std::vector<HTreeTopologyChar>& entries) -> std::optional<HTreeTopologyChar>
{
  if (entries.empty()) {
    return std::nullopt;
  }

  auto pareto_front = BuildDelayPowerParetoFront(entries);
  if (pareto_front.empty()) {
    return entries.front();
  }

  std::ranges::sort(pareto_front, [](const HTreeTopologyChar* lhs, const HTreeTopologyChar* rhs) -> bool {
    LOG_FATAL_IF(lhs == nullptr || rhs == nullptr) << "HTree: null pareto entry encountered during final selection.";
    return PreferPowerMedianOrder(*lhs, *rhs);
  });

  const std::size_t median_index = (pareto_front.size() - 1U) / 2U;
  return *pareto_front.at(median_index);
}

auto BuildLocalDelayPowerPareto(const std::vector<HTreeTopologyChar>& entries) -> std::vector<HTreeTopologyChar>
{
  auto pareto_front = BuildDelayPowerParetoFront(entries);
  std::vector<HTreeTopologyChar> retained_entries;
  retained_entries.reserve(pareto_front.size());
  for (const auto* entry : pareto_front) {
    if (entry != nullptr) {
      retained_entries.push_back(*entry);
    }
  }
  return retained_entries;
}

auto FilterSinkLoadRegionCoveredEntries(const std::vector<HTreeTopologyChar>& entries, const Tree& topology,
                                        const TopologyPatternLibrary& topology_pattern_library,
                                        const BufferPatternLibrary& segment_pattern_library,
                                        SinkLoadRegionLegalityContext& legality_context) -> std::vector<HTreeTopologyChar>
{
  std::vector<HTreeTopologyChar> filtered_entries;
  filtered_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    const auto legality = ResolveSinkLoadRegionLegality(topology, entry.get_pattern_id(), topology_pattern_library, segment_pattern_library,
                                                        legality_context);
    if (!legality.legal) {
      continue;
    }
    if (legality.required_leaf_load_cap_covering_idx.has_value()
        && entry.get_leaf_load_cap_idx() < *legality.required_leaf_load_cap_covering_idx) {
      continue;
    }
    filtered_entries.push_back(entry);
  }
  return filtered_entries;
}

}  // namespace

auto EvaluateCandidateBuild(const std::vector<HTree::LevelPlan>& levels, const SegmentFrontierCatalog& segment_frontier_catalog,
                            const BufferPatternLibrary& segment_pattern_library, const BoundaryConstraints& boundary_constraints,
                            const Tree& topology, SinkLoadRegionLegalityContext& sink_load_region_legality_context, std::size_t leaf_count,
                            unsigned depth, unsigned char_slew_steps, RootDriverCompensationPass& compensation_pass, SchemaWriter& reporter,
                            const HTreeFanoutPruningConfig& fanout_config) -> CandidateBuildEvaluation
{
  CandidateBuildEvaluation result;
  result.depth = depth;
  result.leaf_count = leaf_count;
  result.boundary_constraints = boundary_constraints;
  result.levels = levels;
  compensation_pass.beginCandidateBuild();

  const bool has_boundary_constraints = HasBoundaryConstraints(boundary_constraints);
  const auto topology_assembly = BuildPatternSearch(levels, segment_frontier_catalog, segment_pattern_library, boundary_constraints,
                                                    topology, compensation_pass, reporter, fanout_config);
  if (!topology_assembly.success) {
    result.failure_reason = topology_assembly.failure_reason.empty() ? std::string{"empty_frontier"} : topology_assembly.failure_reason;
    result.failure_level = topology_assembly.failure_level;
    result.failure_length_idx = topology_assembly.failure_length_idx;
    return result;
  }

  result.topology_pattern_library = topology_assembly.topology_pattern_library;
  result.final_frontier_count = topology_assembly.frontier.size();
  result.candidate_solution_count = topology_assembly.frontier.size();
  if (has_boundary_constraints) {
    SinkLoadRegionEntryFilterBuild candidate_sink_load_region_filter;
    SinkLoadRegionEntryFilterBuild feasible_sink_load_region_filter;
    std::vector<HTreeTopologyChar> feasible_raw_frontier;
    {
      auto filter_stage = reporter.beginStage("HTreeDepth", "Filter sink-load region",
                                              {
                                                  {"depth", std::to_string(depth)},
                                                  {"raw_frontier_entries", std::to_string(topology_assembly.frontier.size())},
                                                  {"has_boundary_constraints", "true"},
                                              },
                                              StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail});
      candidate_sink_load_region_filter
          = FilterSinkLoadRegionLegalEntries(topology_assembly.frontier, topology, result.topology_pattern_library, segment_pattern_library,
                                             sink_load_region_legality_context);
      result.feasible_solution_count = CountBoundaryFeasibleHTreeChars(topology_assembly.frontier, boundary_constraints);
      feasible_raw_frontier = FilterBoundaryFeasibleHTreeChars(topology_assembly.frontier, boundary_constraints);
      feasible_sink_load_region_filter = FilterSinkLoadRegionLegalEntries(feasible_raw_frontier, topology, result.topology_pattern_library,
                                                                          segment_pattern_library, sink_load_region_legality_context);
      filter_stage.finished({
          {"candidate_frontier_entries", std::to_string(candidate_sink_load_region_filter.output.entries.size())},
          {"feasible_raw_entries", std::to_string(feasible_raw_frontier.size())},
          {"feasible_frontier_entries", std::to_string(feasible_sink_load_region_filter.output.entries.size())},
      });
    }
    result.candidate_frontier_entries = std::move(candidate_sink_load_region_filter.output.entries);
    result.feasible_frontier_entries = std::move(feasible_sink_load_region_filter.output.entries);
    if (result.candidate_frontier_entries.empty() && !candidate_sink_load_region_filter.summary.first_failure_reason.empty()) {
      result.failure_reason = candidate_sink_load_region_filter.summary.first_failure_reason;
    }
    if (result.feasible_frontier_entries.empty() && result.failure_reason.empty()
        && !feasible_sink_load_region_filter.summary.first_failure_reason.empty()) {
      result.failure_reason = feasible_sink_load_region_filter.summary.first_failure_reason;
    }
  } else {
    result.feasible_solution_count = result.candidate_solution_count;
    SinkLoadRegionEntryFilterBuild feasible_sink_load_region_filter;
    {
      auto filter_stage = reporter.beginStage("HTreeDepth", "Filter sink-load region",
                                              {
                                                  {"depth", std::to_string(depth)},
                                                  {"raw_frontier_entries", std::to_string(topology_assembly.frontier.size())},
                                                  {"has_boundary_constraints", "false"},
                                              },
                                              StageReportOptions{.context_sink = ReportSink::kDetail, .summary_sink = ReportSink::kDetail});
      feasible_sink_load_region_filter
          = FilterSinkLoadRegionLegalEntries(topology_assembly.frontier, topology, result.topology_pattern_library, segment_pattern_library,
                                             sink_load_region_legality_context);
      filter_stage.finished({{"feasible_frontier_entries", std::to_string(feasible_sink_load_region_filter.output.entries.size())}});
    }
    result.feasible_frontier_entries = std::move(feasible_sink_load_region_filter.output.entries);
    if (result.feasible_frontier_entries.empty() && !feasible_sink_load_region_filter.summary.first_failure_reason.empty()) {
      result.failure_reason = feasible_sink_load_region_filter.summary.first_failure_reason;
    }
  }
  if (!result.feasible_frontier_entries.empty()) {
    result.best_char = SelectBestHTreeChar(result.feasible_frontier_entries);
  } else if (has_boundary_constraints && fanout_config.allow_boundary_relaxation) {
    result.best_char = SelectBestHTreeChar(result.candidate_frontier_entries);
    if (result.best_char.has_value()) {
      result.used_boundary_relaxation = true;
      result.boundary_relaxation_reason = "no_strict_boundary_feasible_solution";
      result.boundary_relaxation_score = CalcBoundaryRelaxationScore(*result.best_char, boundary_constraints, char_slew_steps);
    }
  }

  result.success = result.best_char.has_value();
  if (!result.success && result.failure_reason.empty()) {
    result.failure_reason = has_boundary_constraints && !fanout_config.allow_boundary_relaxation
                                ? "no_strict_boundary_feasible_solution"
                                : "no_sink_load_region_legal_frontier_entries";
  }
  return result;
}

auto ReduceCandidateBuildEvaluationForGlobalSelection(CandidateBuildEvaluation& evaluation, const Tree& topology,
                                                      const BufferPatternLibrary& segment_pattern_library,
                                                      SinkLoadRegionLegalityContext& legality_context, bool retain_relaxed_candidates)
    -> void
{
  evaluation.feasible_frontier_entries = BuildLocalDelayPowerPareto(FilterSinkLoadRegionCoveredEntries(
      evaluation.feasible_frontier_entries, topology, evaluation.topology_pattern_library, segment_pattern_library, legality_context));
  if (retain_relaxed_candidates) {
    evaluation.candidate_frontier_entries = BuildLocalDelayPowerPareto(FilterSinkLoadRegionCoveredEntries(
        evaluation.candidate_frontier_entries, topology, evaluation.topology_pattern_library, segment_pattern_library, legality_context));
  } else {
    std::vector<HTreeTopologyChar>().swap(evaluation.candidate_frontier_entries);
  }
  CompactCandidateEvaluationTopologyPatterns(evaluation);
}

}  // namespace icts::htree
