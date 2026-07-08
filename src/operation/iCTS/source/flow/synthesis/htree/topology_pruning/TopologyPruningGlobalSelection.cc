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
 * @file TopologyPruningGlobalSelection.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Global H-tree candidate Pareto selection and coverage filtering.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "PatternId.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts {
class Tree;

namespace htree {
struct BufferPatternLibrary;
}  // namespace htree
}  // namespace icts

namespace icts::htree {
namespace {

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

auto BuildDelayPowerParetoFront(const std::vector<CandidateCharRef>& entries) -> std::vector<CandidateCharRef>
{
  std::vector<CandidateCharRef> sorted_entries;
  sorted_entries.reserve(entries.size());
  for (const auto& entry_ref : entries) {
    if (entry_ref.entry == nullptr) {
      continue;
    }
    sorted_entries.push_back(entry_ref);
  }

  std::ranges::sort(sorted_entries, [](const CandidateCharRef& lhs, const CandidateCharRef& rhs) -> bool {
    LOG_FATAL_IF(lhs.entry == nullptr || rhs.entry == nullptr) << "HTree: null global candidate frontier entry encountered.";
    if (lhs.entry->get_delay() != rhs.entry->get_delay()) {
      return lhs.entry->get_delay() < rhs.entry->get_delay();
    }
    if (lhs.entry->get_power() != rhs.entry->get_power()) {
      return lhs.entry->get_power() < rhs.entry->get_power();
    }
    return PreferPowerMedianOrder(*lhs.entry, *rhs.entry);
  });

  std::vector<CandidateCharRef> pareto_front;
  pareto_front.reserve(sorted_entries.size());
  double best_power_before_delay = std::numeric_limits<double>::infinity();
  std::size_t delay_group_begin = 0U;
  while (delay_group_begin < sorted_entries.size()) {
    std::size_t delay_group_end = delay_group_begin + 1U;
    while (delay_group_end < sorted_entries.size()
           && sorted_entries.at(delay_group_end).entry->get_delay() == sorted_entries.at(delay_group_begin).entry->get_delay()) {
      ++delay_group_end;
    }

    double delay_group_min_power = std::numeric_limits<double>::infinity();
    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      delay_group_min_power = std::min(delay_group_min_power, sorted_entries.at(index).entry->get_power());
    }

    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      const auto& entry_ref = sorted_entries.at(index);
      if (best_power_before_delay <= entry_ref.entry->get_power() || delay_group_min_power < entry_ref.entry->get_power()) {
        continue;
      }
      pareto_front.push_back(entry_ref);
    }

    best_power_before_delay = std::min(best_power_before_delay, delay_group_min_power);
    delay_group_begin = delay_group_end;
  }
  std::ranges::sort(pareto_front, [](const CandidateCharRef& lhs, const CandidateCharRef& rhs) -> bool {
    LOG_FATAL_IF(lhs.entry == nullptr || rhs.entry == nullptr) << "HTree: null global candidate frontier entry encountered.";
    return PreferPowerMedianOrder(*lhs.entry, *rhs.entry);
  });
  return pareto_front;
}

}  // namespace

auto CalcBoundaryRelaxationScore(const HTreeTopologyChar& entry, const BoundaryConstraints& boundary_constraints, unsigned slew_steps)
    -> double
{
  double score = 0.0;
  if (boundary_constraints.top_input_slew_covering_idx.has_value() && slew_steps > 0U) {
    score += static_cast<double>(entry.get_input_slew_idx()) / static_cast<double>(slew_steps);
  }
  return score;
}

auto BuildPerDepthDelayPowerParetoRefs(const std::vector<CandidateCharRef>& entries) -> std::vector<CandidateCharRef>
{
  std::vector<CandidateCharRef> pareto_entries;
  pareto_entries.reserve(entries.size());

  std::unordered_map<std::size_t, std::vector<CandidateCharRef>> entries_by_candidate;
  std::vector<std::size_t> candidate_indices;
  for (const auto& entry_ref : entries) {
    if (entry_ref.entry == nullptr) {
      continue;
    }
    if (!entries_by_candidate.contains(entry_ref.candidate_index)) {
      candidate_indices.push_back(entry_ref.candidate_index);
    }
    entries_by_candidate[entry_ref.candidate_index].push_back(entry_ref);
  }

  for (const auto candidate_index : candidate_indices) {
    const auto group_it = entries_by_candidate.find(candidate_index);
    LOG_FATAL_IF(group_it == entries_by_candidate.end()) << "HTree: missing per-depth candidate group during Pareto compression.";
    auto local_pareto = BuildDelayPowerParetoFront(group_it->second);
    pareto_entries.insert(pareto_entries.end(), local_pareto.begin(), local_pareto.end());
  }
  return pareto_entries;
}

auto SelectBestGlobalEntry(const std::vector<CandidateCharRef>& entries) -> std::optional<CandidateCharRef>
{
  if (entries.empty()) {
    return std::nullopt;
  }

  auto pareto_front = BuildDelayPowerParetoFront(entries);
  if (pareto_front.empty()) {
    return std::nullopt;
  }

  const std::size_t median_index = (pareto_front.size() - 1U) / 2U;
  return pareto_front.at(median_index);
}

auto FilterGlobalEntriesBySinkLoadRegionCoverage(const std::vector<CandidateCharRef>& entries,
                                                 const std::vector<CandidateBuildEvaluation>& evaluations, const Tree& topology,
                                                 const BufferPatternLibrary& segment_pattern_library,
                                                 SinkLoadRegionLegalityContext& legality_context) -> CandidateCharRefFilterBuild
{
  CandidateCharRefFilterBuild result;
  result.output.entries.reserve(entries.size());
  for (const auto& entry_ref : entries) {
    if (entry_ref.entry == nullptr || entry_ref.candidate_index >= evaluations.size()) {
      if (result.summary.first_failure_reason.empty()) {
        result.summary.first_failure_reason = "invalid_global_candidate_ref";
      }
      continue;
    }

    const auto& evaluation = evaluations[entry_ref.candidate_index];
    const auto legality = ResolveSinkLoadRegionLegality(topology, entry_ref.entry->get_pattern_id(), evaluation.topology_pattern_library,
                                                        segment_pattern_library, legality_context);
    if (!legality.legal) {
      if (result.summary.first_failure_reason.empty()) {
        result.summary.first_failure_reason = legality.failure_reason;
      }
      continue;
    }

    if (legality.required_leaf_load_cap_covering_idx.has_value()
        && entry_ref.entry->get_leaf_load_cap_idx() < *legality.required_leaf_load_cap_covering_idx) {
      if (result.summary.first_failure_reason.empty()) {
        std::ostringstream detail;
        detail << "sink_load_region_boundary_load_coverage_violation required_leaf_load_cap_idx="
               << *legality.required_leaf_load_cap_covering_idx << ", entry_leaf_load_cap_idx=" << entry_ref.entry->get_leaf_load_cap_idx()
               << ", max_real_load_cap_pf=" << legality.required_leaf_load_cap_pf;
        result.summary.first_failure_reason = detail.str();
      }
      continue;
    }

    result.output.entries.push_back(entry_ref);
  }
  return result;
}

}  // namespace icts::htree
