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
 * @file SegmentPruning.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Segment frontier synthesis for required H-tree level lengths.
 */

#include "synthesis/htree/segment_pruning/SegmentPruning.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "PatternId.hh"
#include "SegmentChar.hh"
#include "characterization/Characterization.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"

namespace icts::htree {
namespace {

auto resolveSegmentCompositionState(const BufferPatternLibrary& pattern_library, PatternId pattern_id) -> PatternCompositionState
{
  return pattern_library.getCompositionState(pattern_id);
}

auto BuildSegmentStateFrontier(const std::vector<SegmentChar>& chars, const BufferPatternLibrary& pattern_library)
    -> std::vector<SegmentChar>
{
  return icts::BuildSegmentStateFrontier(chars, [&](const SegmentChar& entry) -> PatternCompositionState {
    return resolveSegmentCompositionState(pattern_library, entry.get_pattern_id());
  });
}

auto FindNextSegmentPatternId(const std::vector<SegmentChar>& chars) -> unsigned
{
  unsigned next_id = 0U;
  for (const auto& entry : chars) {
    next_id = std::max(next_id, entry.get_pattern_id().local_id + 1U);
  }
  return next_id;
}

auto hasTerminalBranchBufferPattern(const BufferPatternLibrary& pattern_library, PatternId pattern_id) -> bool
{
  const auto* pattern = pattern_library.find(pattern_id);
  return pattern != nullptr && pattern->hasTerminalBranchBuffer();
}

auto CountSegmentCandidateFrontierEntries(const SegmentCandidateFrontierSet& entry_set) -> std::size_t
{
  return entry_set.countEntries(SegmentFrontierKindSet::full());
}

auto CountTotalSegmentCandidateFrontierEntries(const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& entry_sets) -> std::size_t
{
  std::size_t total_entries = 0U;
  for (const auto& [length_idx, entry_set] : entry_sets) {
    (void) length_idx;
    total_entries += CountSegmentCandidateFrontierEntries(entry_set);
  }
  return total_entries;
}

auto AppendRetainedSegmentPatternIds(const SegmentCandidateFrontierSet& entry_set, SegmentFrontierKindSet required_kinds,
                                     std::vector<PatternId>& pattern_ids) -> void
{
  static constexpr std::array<SegmentFrontierKind, 3> frontier_kinds
      = {SegmentFrontierKind::kAll, SegmentFrontierKind::kTerminalBranchBuffered, SegmentFrontierKind::kTerminalLeafUnbuffered};
  for (const auto kind : frontier_kinds) {
    if (!required_kinds.contains(kind)) {
      continue;
    }
    const auto* entries = entry_set.find(kind);
    if (entries == nullptr) {
      continue;
    }
    pattern_ids.reserve(pattern_ids.size() + entries->size());
    for (const auto& entry : *entries) {
      pattern_ids.push_back(entry.get_pattern_id());
    }
  }
}

auto RetainSegmentPatternsForEntrySets(const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& entry_sets,
                                       SegmentFrontierKindSet required_kinds, BufferPatternLibrary& pattern_library) -> void
{
  std::vector<PatternId> retained_pattern_ids;
  for (const auto& [length_idx, entry_set] : entry_sets) {
    (void) length_idx;
    AppendRetainedSegmentPatternIds(entry_set, required_kinds, retained_pattern_ids);
  }
  pattern_library.retainOnly(retained_pattern_ids);
}

auto FindSegmentCandidateFrontierSet(const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& entry_sets, unsigned length_idx)
    -> const SegmentCandidateFrontierSet*
{
  const auto it = entry_sets.find(length_idx);
  return it == entry_sets.end() ? nullptr : &it->second;
}

auto ComposeSegmentCandidateFrontierEntries(const std::vector<SegmentChar>& upstream, const std::vector<SegmentChar>& downstream,
                                            BufferPatternLibrary& pattern_library, unsigned start_pattern_id)
    -> std::pair<std::vector<SegmentChar>, unsigned>
{
  if (upstream.empty() || downstream.empty()) {
    return {{}, start_pattern_id};
  }

  SegmentPatternLibraryCombiner combiner(pattern_library, start_pattern_id);
  auto pruner = MakeSegmentStateFrontierPruner(
      [&](const SegmentChar& entry) -> PatternCompositionState { return pattern_library.getCompositionState(entry.get_pattern_id()); });
  std::vector<SegmentChar> frontier_entries;
  detail::HashJoinConcat<SegmentChar, SegmentTraits>(upstream, downstream, combiner, frontier_entries, &pruner);
  SortSegmentFrontierEntries(frontier_entries);
  return {std::move(frontier_entries), combiner.get_next_id()};
}

auto ComposeSegmentCandidateFrontierSet(const SegmentCandidateFrontierSet& upstream, const SegmentCandidateFrontierSet& downstream,
                                        BufferPatternLibrary& pattern_library, unsigned start_pattern_id,
                                        SegmentFrontierKindSet required_kinds) -> std::pair<SegmentCandidateFrontierSet, unsigned>
{
  SegmentCandidateFrontierSet result;
  unsigned next_pattern_id = start_pattern_id;

  if (required_kinds.contains(SegmentFrontierKind::kAll)) {
    auto [all_frontier_entries, after_all_pattern_id] = ComposeSegmentCandidateFrontierEntries(
        upstream.require(SegmentFrontierKind::kAll), downstream.require(SegmentFrontierKind::kAll), pattern_library, next_pattern_id);
    result.mutableEntries(SegmentFrontierKind::kAll) = std::move(all_frontier_entries);
    next_pattern_id = after_all_pattern_id;
  }

  if (required_kinds.contains(SegmentFrontierKind::kTerminalBranchBuffered)) {
    auto [branch_frontier_entries, after_branch_pattern_id] = ComposeSegmentCandidateFrontierEntries(
        upstream.require(SegmentFrontierKind::kAll), downstream.require(SegmentFrontierKind::kTerminalBranchBuffered), pattern_library,
        next_pattern_id);
    result.mutableEntries(SegmentFrontierKind::kTerminalBranchBuffered) = std::move(branch_frontier_entries);
    next_pattern_id = after_branch_pattern_id;
  }

  if (required_kinds.contains(SegmentFrontierKind::kTerminalLeafUnbuffered)) {
    auto [leaf_frontier_entries, after_leaf_pattern_id] = ComposeSegmentCandidateFrontierEntries(
        upstream.require(SegmentFrontierKind::kAll), downstream.require(SegmentFrontierKind::kTerminalLeafUnbuffered), pattern_library,
        next_pattern_id);
    result.mutableEntries(SegmentFrontierKind::kTerminalLeafUnbuffered) = std::move(leaf_frontier_entries);
    next_pattern_id = after_leaf_pattern_id;
  }

  return {std::move(result), next_pattern_id};
}

auto BuildBaseSegmentCandidateLengthEntrySets(const std::vector<SegmentChar>& chars, const BufferPatternLibrary& pattern_library,
                                              SegmentFrontierKindSet required_kinds)
    -> std::unordered_map<unsigned, SegmentCandidateFrontierSet>
{
  std::unordered_map<unsigned, std::vector<SegmentChar>> raw_all_by_length;
  std::unordered_map<unsigned, std::vector<SegmentChar>> raw_leaf_unbuffered_by_length;
  std::unordered_map<unsigned, std::vector<SegmentChar>> raw_branch_by_length;
  const bool build_all = required_kinds.contains(SegmentFrontierKind::kAll);
  const bool build_branch = required_kinds.contains(SegmentFrontierKind::kTerminalBranchBuffered);
  const bool build_leaf = required_kinds.contains(SegmentFrontierKind::kTerminalLeafUnbuffered);
  raw_all_by_length.reserve(chars.size());
  if (build_leaf) {
    raw_leaf_unbuffered_by_length.reserve(chars.size());
  }
  if (build_branch) {
    raw_branch_by_length.reserve(chars.size());
  }
  for (const auto& entry : chars) {
    if (build_all) {
      raw_all_by_length[entry.get_length_idx()].push_back(entry);
    }
    if (!build_branch && !build_leaf) {
      continue;
    }
    if (hasTerminalBranchBufferPattern(pattern_library, entry.get_pattern_id())) {
      if (build_branch) {
        raw_branch_by_length[entry.get_length_idx()].push_back(entry);
      }
    } else if (build_leaf) {
      raw_leaf_unbuffered_by_length[entry.get_length_idx()].push_back(entry);
    }
  }

  std::unordered_map<unsigned, SegmentCandidateFrontierSet> entry_sets_by_length;
  entry_sets_by_length.reserve(raw_all_by_length.size());
  for (auto& [length_idx, raw_entries] : raw_all_by_length) {
    auto& entry_set = entry_sets_by_length[length_idx];
    entry_set.mutableEntries(SegmentFrontierKind::kAll) = BuildSegmentStateFrontier(raw_entries, pattern_library);
  }

  for (auto& [length_idx, entry_set] : entry_sets_by_length) {
    (void) length_idx;
    if (build_branch) {
      (void) entry_set.mutableEntries(SegmentFrontierKind::kTerminalBranchBuffered);
    }
    if (build_leaf) {
      (void) entry_set.mutableEntries(SegmentFrontierKind::kTerminalLeafUnbuffered);
    }
  }

  if (build_branch) {
    for (auto& [length_idx, raw_entries] : raw_branch_by_length) {
      auto& entry_set = entry_sets_by_length[length_idx];
      entry_set.mutableEntries(SegmentFrontierKind::kTerminalBranchBuffered) = BuildSegmentStateFrontier(raw_entries, pattern_library);
    }
  }
  if (build_leaf) {
    for (auto& [length_idx, raw_entries] : raw_leaf_unbuffered_by_length) {
      auto& entry_set = entry_sets_by_length[length_idx];
      entry_set.mutableEntries(SegmentFrontierKind::kTerminalLeafUnbuffered) = BuildSegmentStateFrontier(raw_entries, pattern_library);
    }
  }
  return entry_sets_by_length;
}

auto NormalizeRequiredLengths(std::vector<unsigned> lengths) -> std::vector<unsigned>
{
  std::erase(lengths, 0U);
  std::ranges::sort(lengths);
  const auto unique_tail = std::ranges::unique(lengths);
  lengths.erase(unique_tail.begin(), unique_tail.end());
  return lengths;
}

auto BuildPendingLengthKey(const std::vector<unsigned>& pending_lengths,
                           const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& base_entry_sets) -> RequiredLengthStateKey
{
  std::vector<unsigned> canonical_lengths;
  canonical_lengths.reserve(pending_lengths.size());
  for (const unsigned length_idx : pending_lengths) {
    const auto* base_entry_set = FindSegmentCandidateFrontierSet(base_entry_sets, length_idx);
    const auto* all_frontier = base_entry_set == nullptr ? nullptr : base_entry_set->find(SegmentFrontierKind::kAll);
    if (all_frontier != nullptr && !all_frontier->empty()) {
      continue;
    }
    canonical_lengths.push_back(length_idx);
  }

  return RequiredLengthStateKey{.pending_lengths = NormalizeRequiredLengths(std::move(canonical_lengths))};
}

auto ResolveSegmentCandidateFrontierSet(unsigned length_idx,
                                        const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& base_entry_sets,
                                        const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& synthesized_entry_sets)
    -> const SegmentCandidateFrontierSet*
{
  if (const auto* synthesized_entry_set = FindSegmentCandidateFrontierSet(synthesized_entry_sets, length_idx);
      synthesized_entry_set != nullptr) {
    return synthesized_entry_set;
  }
  return FindSegmentCandidateFrontierSet(base_entry_sets, length_idx);
}

auto PreferSegmentClosureSolution(const SegmentClosureSolution& lhs, const SegmentClosureSolution& rhs) -> bool
{
  if (!lhs.feasible) {
    return false;
  }
  if (!rhs.feasible) {
    return true;
  }
  if (lhs.total_cost != rhs.total_cost) {
    return lhs.total_cost < rhs.total_cost;
  }

  const std::size_t lhs_frontier_entries = CountTotalSegmentCandidateFrontierEntries(lhs.synthesized_entry_sets);
  const std::size_t rhs_frontier_entries = CountTotalSegmentCandidateFrontierEntries(rhs.synthesized_entry_sets);
  if (lhs_frontier_entries != rhs_frontier_entries) {
    return lhs_frontier_entries > rhs_frontier_entries;
  }

  return lhs.synthesized_entry_sets.size() < rhs.synthesized_entry_sets.size();
}

auto SolveRequiredLengthState(const RequiredLengthStateKey& state_key,
                              const std::unordered_map<unsigned, SegmentCandidateFrontierSet>& base_entry_sets,
                              BufferPatternLibrary& pattern_library, unsigned& next_pattern_id,
                              std::unordered_map<RequiredLengthStateKey, SegmentClosureSolution, RequiredLengthStateKeyHash>& memo,
                              SegmentFrontierKindSet required_kinds) -> SegmentClosureSolution
{
  std::vector<RequiredLengthStateKey> pending_states = {state_key};
  while (!pending_states.empty()) {
    const auto current_key = pending_states.back();
    if (memo.contains(current_key)) {
      pending_states.pop_back();
      continue;
    }
    if (current_key.pending_lengths.empty()) {
      memo[current_key] = SegmentClosureSolution{
          .feasible = true,
          .total_cost = 0U,
          .synthesized_entry_sets = {},
      };
      pending_states.pop_back();
      continue;
    }

    const unsigned target_length_idx = current_key.pending_lengths.back();
    std::vector<unsigned> remaining_lengths = current_key.pending_lengths;
    remaining_lengths.pop_back();

    std::vector<RequiredLengthStateKey> unresolved_sub_states;
    for (unsigned left_length_idx = 1U; left_length_idx < target_length_idx; ++left_length_idx) {
      const unsigned right_length_idx = target_length_idx - left_length_idx;
      auto sub_pending_lengths = remaining_lengths;
      sub_pending_lengths.push_back(left_length_idx);
      sub_pending_lengths.push_back(right_length_idx);
      const RequiredLengthStateKey sub_state_key = BuildPendingLengthKey(sub_pending_lengths, base_entry_sets);
      if (!memo.contains(sub_state_key)) {
        unresolved_sub_states.push_back(sub_state_key);
      }
    }
    if (!unresolved_sub_states.empty()) {
      for (const auto& sub_state_key : std::views::reverse(unresolved_sub_states)) {
        pending_states.push_back(sub_state_key);
      }
      continue;
    }

    SegmentClosureSolution best_solution;
    for (unsigned left_length_idx = 1U; left_length_idx < target_length_idx; ++left_length_idx) {
      const unsigned right_length_idx = target_length_idx - left_length_idx;

      auto sub_pending_lengths = remaining_lengths;
      sub_pending_lengths.push_back(left_length_idx);
      sub_pending_lengths.push_back(right_length_idx);
      const RequiredLengthStateKey sub_state_key = BuildPendingLengthKey(sub_pending_lengths, base_entry_sets);
      const auto& sub_solution = memo.at(sub_state_key);
      if (!sub_solution.feasible) {
        continue;
      }

      const auto* left_entry_set
          = ResolveSegmentCandidateFrontierSet(left_length_idx, base_entry_sets, sub_solution.synthesized_entry_sets);
      const auto* right_entry_set
          = ResolveSegmentCandidateFrontierSet(right_length_idx, base_entry_sets, sub_solution.synthesized_entry_sets);
      if (left_entry_set == nullptr || right_entry_set == nullptr) {
        continue;
      }
      const auto* left_all_frontier = left_entry_set->find(SegmentFrontierKind::kAll);
      const auto* right_all_frontier = right_entry_set->find(SegmentFrontierKind::kAll);
      if (left_all_frontier == nullptr || right_all_frontier == nullptr || left_all_frontier->empty() || right_all_frontier->empty()) {
        continue;
      }

      auto [composed_entry_set, updated_next_pattern_id]
          = ComposeSegmentCandidateFrontierSet(*left_entry_set, *right_entry_set, pattern_library, next_pattern_id, required_kinds);
      next_pattern_id = updated_next_pattern_id;
      const auto* composed_all_frontier = composed_entry_set.find(SegmentFrontierKind::kAll);
      if (composed_all_frontier == nullptr || composed_all_frontier->empty()) {
        continue;
      }

      auto candidate_solution = sub_solution;
      candidate_solution.feasible = true;
      candidate_solution.total_cost += target_length_idx;
      candidate_solution.synthesized_entry_sets[target_length_idx] = std::move(composed_entry_set);
      if (PreferSegmentClosureSolution(candidate_solution, best_solution)) {
        best_solution = std::move(candidate_solution);
      }
    }

    memo[current_key] = std::move(best_solution);
    pending_states.pop_back();
  }

  return memo.at(state_key);
}

auto SynthesizeSegmentFrontierSets(const std::vector<SegmentChar>& base_segment_chars, BufferPatternLibrary& pattern_library,
                                   const RequiredSegmentFrontiers& required_frontiers)
    -> std::unordered_map<unsigned, SegmentCandidateFrontierSet>
{
  const SegmentFrontierKindSet required_kinds = required_frontiers.required_kinds.normalized();
  if (required_kinds.empty()) {
    return {};
  }

  auto entry_sets_by_length = BuildBaseSegmentCandidateLengthEntrySets(base_segment_chars, pattern_library, required_kinds);
  const RequiredLengthStateKey root_state_key = BuildPendingLengthKey(required_frontiers.required_length_indices, entry_sets_by_length);
  if (root_state_key.pending_lengths.empty()) {
    RetainSegmentPatternsForEntrySets(entry_sets_by_length, required_kinds, pattern_library);
    return entry_sets_by_length;
  }

  unsigned next_pattern_id = FindNextSegmentPatternId(base_segment_chars);
  std::unordered_map<RequiredLengthStateKey, SegmentClosureSolution, RequiredLengthStateKeyHash> memo;
  auto closure_solution
      = SolveRequiredLengthState(root_state_key, entry_sets_by_length, pattern_library, next_pattern_id, memo, required_kinds);
  if (!closure_solution.feasible) {
    return {};
  }

  for (auto& [length_idx, entry_set] : closure_solution.synthesized_entry_sets) {
    entry_sets_by_length[length_idx] = std::move(entry_set);
  }
  RetainSegmentPatternsForEntrySets(entry_sets_by_length, required_kinds, pattern_library);
  return entry_sets_by_length;
}

}  // namespace

auto CollectRequiredLengthIndices(const std::vector<HTree::LevelPlan>& levels) -> std::vector<unsigned>
{
  std::vector<unsigned> required_lengths;
  required_lengths.reserve(levels.size());
  for (const auto& level : levels) {
    if (level.aligned_length_idx > 0U) {
      required_lengths.push_back(level.aligned_length_idx);
    }
  }
  return NormalizeRequiredLengths(std::move(required_lengths));
}

auto ResolveRequiredSegmentFrontiers(std::vector<unsigned> required_length_indices, const BoundaryConstraints& boundary_constraints)
    -> RequiredSegmentFrontiers
{
  return RequiredSegmentFrontiers{
      .required_length_indices = std::move(required_length_indices),
      .required_kinds
      = boundary_constraints.force_branch_buffer ? SegmentFrontierKindSet::branchConstrained() : SegmentFrontierKindSet::allOnly(),
  };
}

auto SynthesizeSegmentFrontiers(const std::vector<SegmentChar>& base_segment_chars, BufferPatternLibrary& pattern_library,
                                const RequiredSegmentFrontiers& required_frontiers) -> SegmentFrontierCatalog
{
  return SegmentFrontierCatalog(SynthesizeSegmentFrontierSets(base_segment_chars, pattern_library, required_frontiers));
}

}  // namespace icts::htree
