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
 * @file AnalyticalSolveProblem.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Analytical H-tree solve-problem validation and diagnostics.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AnalyticalModel.hh"
#include "BufferingPattern.hh"
#include "PatternId.hh"
#include "ValueLattice.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/analytical_solver/model/AnalyticalSolverModel.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"

namespace icts::htree::analytical_solver {

auto MakeFailure(std::string reason) -> AnalyticalSolverBuild
{
  AnalyticalSolverBuild result;
  result.summary.success = false;
  result.summary.failure_reason = std::move(reason);
  return result;
}

auto ValidateSolveProblem(const AnalyticalHTreeSolveProblem& solve_problem) -> std::string
{
  if (solve_problem.levels == nullptr || solve_problem.levels->empty()) {
    return "missing_levels";
  }
  if (solve_problem.segment_pattern_library == nullptr) {
    return "missing_segment_pattern_library";
  }
  if (solve_problem.config.use_functional_unit_compose && solve_problem.mutable_segment_pattern_library == nullptr) {
    return "missing_mutable_segment_pattern_library";
  }
  if (solve_problem.model_catalog == nullptr || solve_problem.model_catalog->empty()) {
    return "missing_analytical_model_catalog";
  }
  if (!solve_problem.slew_lattice.isValid() || !solve_problem.cap_lattice.isValid()) {
    return "invalid_lattice";
  }
  if (solve_problem.config.root_input_slew_ns < 0.0) {
    return "invalid_root_input_slew";
  }
  if (solve_problem.config.representative_leaf_load_cap_pf <= 0.0) {
    return "invalid_representative_leaf_load_cap";
  }
  return {};
}

auto ResolveNextSegmentPatternId(const BufferPatternLibrary& segment_pattern_library) -> unsigned
{
  unsigned next_id = 0U;
  for (const auto& [pattern_id, pattern] : segment_pattern_library.patterns) {
    (void) pattern;
    if (pattern_id.domain == PatternDomain::kSegmentPattern) {
      next_id = std::max(next_id, pattern_id.local_id + 1U);
    }
  }
  return next_id;
}

auto MakePatternSequenceKey(const std::vector<PatternId>& pattern_ids) -> PatternSequenceKey
{
  PatternSequenceKey key;
  key.pattern_ids.reserve(pattern_ids.size());
  for (const auto pattern_id : pattern_ids) {
    key.pattern_ids.push_back(pattern_id.pack());
  }
  return key;
}

auto ResolveSegmentPatternLibrary(const AnalyticalHTreeSolveProblem& solve_problem) -> const BufferPatternLibrary*
{
  return solve_problem.segment_pattern_library != nullptr ? solve_problem.segment_pattern_library
                                                          : solve_problem.mutable_segment_pattern_library;
}

auto MaterializeFunctionalSegmentPattern(const std::vector<PatternId>& unit_pattern_ids, FunctionalComposeContext& context,
                                         BufferPatternLibrary& segment_pattern_library) -> std::optional<PatternId>
{
  if (unit_pattern_ids.empty()) {
    return std::nullopt;
  }
  if (unit_pattern_ids.size() == 1U) {
    return unit_pattern_ids.front();
  }

  const auto key = MakePatternSequenceKey(unit_pattern_ids);
  if (const auto it = context.materialized_patterns.find(key); it != context.materialized_patterns.end()) {
    return it->second;
  }

  if (segment_pattern_library.find(unit_pattern_ids.front()) == nullptr) {
    return std::nullopt;
  }
  SegmentPatternLibraryCombiner combiner(segment_pattern_library, context.next_segment_pattern_id);
  PatternId merged_pattern_id = unit_pattern_ids.front();
  for (std::size_t index = 1U; index < unit_pattern_ids.size(); ++index) {
    const PatternId next_pattern_id = unit_pattern_ids.at(index);
    if (segment_pattern_library.find(next_pattern_id) == nullptr || !combiner.canCompose(merged_pattern_id, next_pattern_id)) {
      return std::nullopt;
    }
    merged_pattern_id = combiner.combine(merged_pattern_id, next_pattern_id);
  }

  context.next_segment_pattern_id = combiner.get_next_id();
  context.materialized_patterns.emplace(key, merged_pattern_id);
  return merged_pattern_id;
}

auto MakeSegmentChoice(std::size_t level_index, const ScoredSegment& selected) -> AnalyticalSegmentChoice
{
  return AnalyticalSegmentChoice{
      .level_index = level_index,
      .length_idx = selected.length_idx,
      .segment_pattern_id = selected.pattern_id,
      .input_slew_ns = selected.input_slew_ns,
      .downstream_load_cap_pf = selected.downstream_load_cap_pf,
      .output_slew_ns = selected.output_slew_ns,
      .source_cap_pf = selected.source_cap_pf,
      .delay_ns = selected.delay_ns,
      .power_w = selected.power_w,
      .source_boundary_power_w = selected.source_boundary_power_w,
      .slew_upper_ns = selected.slew_upper_ns,
      .delay_upper_ns = selected.delay_upper_ns,
      .power_upper_w = selected.power_upper_w,
  };
}

auto AccumulateHTreePower(double accumulated_power_w, std::size_t level_index, const ScoredSegment& selected) -> double
{
  if (level_index == 0U) {
    return accumulated_power_w + selected.power_w;
  }
  const double level_multiplicity = std::ldexp(1.0, static_cast<int>(level_index));
  return accumulated_power_w + level_multiplicity * (selected.power_w - selected.source_boundary_power_w);
}

}  // namespace icts::htree::analytical_solver
