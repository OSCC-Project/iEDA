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
 * @file AnalyticalCandidate.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Mathematical analytical H-tree candidate helpers.
 */

#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "CharCore.hh"
#include "ValueLattice.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"

namespace icts {
struct PatternCompositionState;
}  // namespace icts

namespace icts::htree::analytical_solver {
namespace {

auto BuildSeedTopologyPattern(TopologyPatternLibrary& library, PatternId topology_pattern_id, PatternId segment_pattern_id,
                              const PatternCompositionState& composition_state) -> void
{
  library.addSeed(topology_pattern_id, segment_pattern_id, composition_state);
}

}  // namespace

auto AnalyticalCandidate::isValid() const -> bool
{
  return rejection_reason.empty() && branch_buffer_legal && fanout_legal && !level_segment_pattern_ids.empty();
}

auto LexicographicalPatternIdLess(const std::vector<PatternId>& lhs, const std::vector<PatternId>& rhs) -> bool
{
  return std::ranges::lexicographical_compare(lhs, rhs, [](PatternId left, PatternId right) -> bool { return left.pack() < right.pack(); });
}

auto PreferAnalyticalCandidate(const AnalyticalCandidate& lhs, const AnalyticalCandidate& rhs) -> bool
{
  if (lhs.conservative_delay_ns != rhs.conservative_delay_ns) {
    return lhs.conservative_delay_ns < rhs.conservative_delay_ns;
  }
  if (lhs.conservative_power_w != rhs.conservative_power_w) {
    return lhs.conservative_power_w < rhs.conservative_power_w;
  }
  if (lhs.root_source_cap_pf != rhs.root_source_cap_pf) {
    return lhs.root_source_cap_pf < rhs.root_source_cap_pf;
  }
  return LexicographicalPatternIdLess(lhs.level_segment_pattern_ids, rhs.level_segment_pattern_ids);
}

auto BuildAnalyticalTopologyPattern(const std::vector<PatternId>& level_segment_pattern_ids,
                                    const BufferPatternLibrary& segment_pattern_library, std::size_t max_fanout)
    -> std::optional<TopologyPatternLibrary>
{
  TopologyPatternLibrary library;
  if (level_segment_pattern_ids.empty()) {
    return std::nullopt;
  }

  unsigned next_pattern_id = 0U;
  PatternId current_pattern_id = PatternId::topology(next_pattern_id++);
  BuildSeedTopologyPattern(library, current_pattern_id, level_segment_pattern_ids.back(),
                           segment_pattern_library.getCompositionState(level_segment_pattern_ids.back()));

  for (std::size_t reverse_index = level_segment_pattern_ids.size() - 1U; reverse_index > 0U; --reverse_index) {
    const PatternId upstream_topology_id = PatternId::topology(next_pattern_id++);
    const PatternId upstream_segment_id = level_segment_pattern_ids.at(reverse_index - 1U);
    BuildSeedTopologyPattern(library, upstream_topology_id, upstream_segment_id,
                             segment_pattern_library.getCompositionState(upstream_segment_id));

    TopologyPatternLibraryCombiner combiner(library, next_pattern_id, max_fanout);
    if (!combiner.canCompose(upstream_topology_id, current_pattern_id)) {
      return std::nullopt;
    }
    current_pattern_id = combiner.combine(upstream_topology_id, current_pattern_id);
    next_pattern_id = combiner.get_next_id();
  }
  return library;
}

auto MaterializeAnalyticalTopologyChar(const AnalyticalCandidate& candidate, const icts::UniformValueLattice& slew_lattice,
                                       const icts::UniformValueLattice& cap_lattice) -> std::optional<HTreeTopologyChar>
{
  if (!candidate.isValid() || !slew_lattice.isValid() || !cap_lattice.isValid()) {
    return std::nullopt;
  }

  const auto input_slew_idx = slew_lattice.tryObservedIndex(std::max(candidate.root_input_slew_ns, slew_lattice.stepValue()));
  const auto output_slew_idx = slew_lattice.tryObservedIndex(std::max(candidate.output_slew_ns, slew_lattice.stepValue()));
  const auto driven_cap_idx = cap_lattice.tryObservedIndex(std::max(candidate.root_source_cap_pf, cap_lattice.stepValue()));
  const auto load_cap_idx = cap_lattice.tryObservedIndex(std::max(candidate.leaf_load_cap_pf, cap_lattice.stepValue()));
  if (!input_slew_idx.has_value() || !output_slew_idx.has_value() || !driven_cap_idx.has_value() || !load_cap_idx.has_value()) {
    return std::nullopt;
  }

  const PatternId topology_pattern_id = PatternId::topology(
      candidate.topology_pattern_library.nodes.empty() ? 0U : static_cast<unsigned>(candidate.topology_pattern_library.nodes.size() - 1U));
  CharCore core(*input_slew_idx, *output_slew_idx, *driven_cap_idx, *load_cap_idx, candidate.raw_delay_ns, candidate.raw_power_w,
                topology_pattern_id, 0.0);
  return HTreeTopologyChar(std::move(core), candidate.depth);
}

}  // namespace icts::htree::analytical_solver
