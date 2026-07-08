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
 * @file AnalyticalSolverModel.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief Support contracts for mathematical analytical H-tree materialization.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "PatternId.hh"
#include "characterization/Characterization.hh"
#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"

namespace icts::analytical {
struct AnalyticalModelSet;
}  // namespace icts::analytical

namespace icts::htree {
struct BufferPatternLibrary;
}  // namespace icts::htree

namespace icts::htree::analytical_solver {

struct ScoredSegment
{
  PatternId pattern_id = PatternId::segment(0U);
  unsigned length_idx = 0U;
  double score = 0.0;
  double input_slew_ns = 0.0;
  double downstream_load_cap_pf = 0.0;
  double output_slew_ns = 0.0;
  double source_cap_pf = 0.0;
  double delay_ns = 0.0;
  double power_w = 0.0;
  double source_boundary_power_w = 0.0;
  double slew_upper_ns = 0.0;
  double delay_upper_ns = 0.0;
  double power_upper_w = 0.0;
  std::vector<PatternId> unit_pattern_ids;
};

struct UnitModelRef
{
  PatternId pattern_id = PatternId::segment(0U);
  const icts::analytical::AnalyticalModelSet* model_set = nullptr;
  PatternCompositionState composition_state;
};

struct PatternSequenceKey
{
  std::vector<unsigned> pattern_ids;

  auto operator==(const PatternSequenceKey& rhs) const -> bool = default;
};

struct PatternSequenceKeyHash
{
  auto operator()(const PatternSequenceKey& key) const noexcept -> std::size_t
  {
    std::size_t hash_value = 0U;
    for (const unsigned pattern_id : key.pattern_ids) {
      hash_value ^= std::hash<unsigned>{}(pattern_id) + 0x9e3779b9U + (hash_value << 6U) + (hash_value >> 2U);
    }
    return hash_value;
  }
};

struct FunctionalComposeContext
{
  std::vector<UnitModelRef> unit_models;
  std::unordered_map<std::string, PatternId> unit_pattern_by_cell_master_and_terminal_semantic;
  std::unordered_map<PatternId, std::vector<PatternId>> decomposed_patterns;
  std::unordered_map<PatternSequenceKey, PatternId, PatternSequenceKeyHash> materialized_patterns;
  unsigned next_segment_pattern_id = 0U;
};

auto MakeFailure(std::string reason) -> AnalyticalSolverBuild;
auto ValidateSolveProblem(const AnalyticalHTreeSolveProblem& solve_problem) -> std::string;
auto ResolveNextSegmentPatternId(const BufferPatternLibrary& segment_pattern_library) -> unsigned;
auto MakePatternSequenceKey(const std::vector<PatternId>& pattern_ids) -> PatternSequenceKey;
auto ResolveSegmentPatternLibrary(const AnalyticalHTreeSolveProblem& solve_problem) -> const BufferPatternLibrary*;
auto MaterializeFunctionalSegmentPattern(const std::vector<PatternId>& unit_pattern_ids, FunctionalComposeContext& context,
                                         BufferPatternLibrary& segment_pattern_library) -> std::optional<PatternId>;
auto ResolveAnalyticalRootProbeSlewNs(const AnalyticalHTreeSolveProblem& solve_problem) -> double;
auto CollectUnitModelRefs(const AnalyticalHTreeSolveProblem& solve_problem) -> std::vector<UnitModelRef>;
auto BuildUnitPatternByCellMaster(const AnalyticalHTreeSolveProblem& solve_problem, const std::vector<UnitModelRef>& unit_models)
    -> std::unordered_map<std::string, PatternId>;
auto ScoreFunctionalUnitSequence(const AnalyticalHTreeSolveProblem& solve_problem, const std::vector<PatternId>& unit_pattern_ids,
                                 PatternId materialized_pattern_id, unsigned length_idx, double input_slew_ns, double downstream_cap_pf,
                                 bool conservative, AnalyticalSolverBuild& result) -> std::optional<ScoredSegment>;
auto MakeSegmentChoice(std::size_t level_index, const ScoredSegment& selected) -> AnalyticalSegmentChoice;
auto AccumulateHTreePower(double accumulated_power_w, std::size_t level_index, const ScoredSegment& selected) -> double;

}  // namespace icts::htree::analytical_solver
