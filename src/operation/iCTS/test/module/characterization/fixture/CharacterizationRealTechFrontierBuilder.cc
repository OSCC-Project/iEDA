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
 * @file CharacterizationRealTechFrontierBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Exact frontier composition helpers for real-tech characterization tests.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "Flow.hh"
#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "SegmentChar.hh"
#include "characterization/Characterization.hh"
#include "common/CTSTestRuntime.hh"
#include "database/characterization/CharCore.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/ValueLattice.hh"
#include "database/io/Wrapper.hh"
#include "module/characterization/fixture/CharacterizationRealTechFixture.hh"

namespace icts_test::characterization::realtech {
namespace {

class BufferStrengthTable
{
 public:
  auto getStrengthRank(const std::string& cell_master) -> unsigned
  {
    if (cell_master.empty()) {
      return 0U;
    }

    if (!_drive_caps.contains(cell_master)) {
      double drive_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapLimit(cell_master);
      if (drive_cap_pf <= 0.0) {
        drive_cap_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCellOutPinCapTableAxisMax(cell_master);
      }
      _drive_caps[cell_master] = drive_cap_pf;
      _ranks_dirty = true;

      if (drive_cap_pf <= 0.0) {
        LOG_WARNING << "CharacterizationRealTechFixture: unresolved drive-strength rank for buffer master " << cell_master
                    << "; manual compose keeps an explicit boundary buffer with unresolved size class.";
      }
    }

    if (_ranks_dirty) {
      rebuildRanks();
    }

    const auto rank_it = _strength_ranks.find(cell_master);
    return rank_it == _strength_ranks.end() ? 0U : rank_it->second;
  }

 private:
  auto rebuildRanks() -> void
  {
    std::vector<std::pair<std::string, double>> ordered_caps;
    ordered_caps.reserve(_drive_caps.size());
    for (const auto& [cell_master, drive_cap_pf] : _drive_caps) {
      if (drive_cap_pf > 0.0) {
        ordered_caps.emplace_back(cell_master, drive_cap_pf);
      }
    }

    std::ranges::sort(ordered_caps, [](const auto& lhs, const auto& rhs) -> bool {
      if (lhs.second != rhs.second) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    });

    _strength_ranks.clear();
    unsigned current_rank = 0U;
    double last_cap_pf = 0.0;
    bool has_last_cap = false;
    for (const auto& [cell_master, drive_cap_pf] : ordered_caps) {
      if (!has_last_cap || std::abs(drive_cap_pf - last_cap_pf) > 1e-12) {
        ++current_rank;
        last_cap_pf = drive_cap_pf;
        has_last_cap = true;
      }
      _strength_ranks[cell_master] = current_rank;
    }

    _ranks_dirty = false;
  }

  std::unordered_map<std::string, double> _drive_caps;
  std::unordered_map<std::string, unsigned> _strength_ranks;
  bool _ranks_dirty = false;
};

auto ResolveBoundaryBufferState(const icts::BoundaryBufferState& explicit_state, const std::string& cell_master,
                                BufferStrengthTable& strength_table) -> icts::BoundaryBufferState
{
  if (explicit_state.has_buffer) {
    return explicit_state;
  }
  if (cell_master.empty()) {
    return explicit_state;
  }
  return icts::BoundaryBufferState{.has_buffer = true, .strength_rank = strength_table.getStrengthRank(cell_master)};
}

auto ResolvePatternBoundaryState(const icts::BufferingPattern& pattern, BufferStrengthTable& strength_table) -> icts::MonotonicBoundaryState
{
  const auto explicit_state = pattern.get_monotonic_boundary_state();
  if (!pattern.isBufferPattern()) {
    return explicit_state;
  }

  const auto& cell_masters = pattern.get_cell_masters();
  if (cell_masters.empty()) {
    return explicit_state;
  }

  return icts::MonotonicBoundaryState{
      .source = ResolveBoundaryBufferState(explicit_state.source, cell_masters.front(), strength_table),
      .sink = ResolveBoundaryBufferState(explicit_state.sink, cell_masters.back(), strength_table),
  };
}

auto EnrichPatternBoundaryState(const icts::BufferingPattern& pattern, BufferStrengthTable& strength_table) -> icts::BufferingPattern
{
  const auto boundary_state = ResolvePatternBoundaryState(pattern, strength_table);
  if (boundary_state == pattern.get_monotonic_boundary_state()) {
    return pattern;
  }

  return icts::BufferingPattern(pattern.get_length_idx(), pattern.get_pattern_id(), pattern.get_buffer_positions(),
                                pattern.get_cell_masters(), pattern.hasTerminalBranchBuffer(), boundary_state);
}

auto BuildCompositionState(const icts::BufferingPattern& pattern) -> icts::PatternCompositionState
{
  return icts::PatternCompositionState{
      .terminal_semantic
      = pattern.hasTerminalBranchBuffer() ? icts::TerminalSemantic::kBranchBuffered : icts::TerminalSemantic::kLeafUnbuffered,
      .monotonic_boundary_state = pattern.get_monotonic_boundary_state(),
  };
}

auto LookupSegmentState(const SegmentFrontierContext& context, icts::PatternId pattern_id) -> icts::PatternCompositionState
{
  const auto it = context.composition_states.find(pattern_id);
  LOG_FATAL_IF(it == context.composition_states.end())
      << "CharacterizationRealTechFixture: missing segment composition state for pattern " << pattern_id.local_id;
  return it->second;
}

auto LookupTopologyState(const HTreeFrontierContext& context, icts::PatternId pattern_id) -> icts::PatternCompositionState
{
  const auto it = context.composition_states.find(pattern_id);
  LOG_FATAL_IF(it == context.composition_states.end())
      << "CharacterizationRealTechFixture: missing topology composition state for pattern " << pattern_id.local_id;
  return it->second;
}

class SegmentStateCombiner
{
 public:
  explicit SegmentStateCombiner(SegmentFrontierContext* context) : _context(context) {}

  auto canCompose(icts::PatternId upstream, icts::PatternId downstream) const -> bool
  {
    const auto upstream_state = LookupSegmentState(*_context, upstream);
    const auto downstream_state = LookupSegmentState(*_context, downstream);
    return upstream_state.monotonic_boundary_state.canComposeWith(downstream_state.monotonic_boundary_state);
  }

  auto combine(icts::PatternId upstream, icts::PatternId downstream) const -> icts::PatternId
  {
    LOG_FATAL_IF(!canCompose(upstream, downstream)) << "CharacterizationRealTechFixture: invalid non-monotonic segment composition.";
    const auto upstream_it = _context->patterns.find(upstream);
    const auto downstream_it = _context->patterns.find(downstream);
    LOG_FATAL_IF(upstream_it == _context->patterns.end() || downstream_it == _context->patterns.end())
        << "CharacterizationRealTechFixture: missing segment pattern during exact compose.";

    const auto merged_pattern_id = icts::PatternId::segment(_context->next_pattern_id++);
    auto merged_pattern = icts::BufferingPattern::concat(upstream_it->second, downstream_it->second);
    merged_pattern = icts::BufferingPattern(merged_pattern.get_length_idx(), merged_pattern_id, merged_pattern.get_buffer_positions(),
                                            merged_pattern.get_cell_masters(), merged_pattern.hasTerminalBranchBuffer(),
                                            merged_pattern.get_monotonic_boundary_state());
    _context->patterns[merged_pattern_id] = merged_pattern;
    _context->composition_states[merged_pattern_id] = BuildCompositionState(merged_pattern);
    return merged_pattern_id;
  }

 private:
  SegmentFrontierContext* _context = nullptr;
};

class TopologyStateCombiner
{
 public:
  explicit TopologyStateCombiner(HTreeFrontierContext* context) : _context(context) {}

  auto canCompose(icts::PatternId upstream, icts::PatternId downstream) const -> bool
  {
    const auto upstream_state = LookupTopologyState(*_context, upstream);
    const auto downstream_state = LookupTopologyState(*_context, downstream);
    return upstream_state.monotonic_boundary_state.canComposeWith(downstream_state.monotonic_boundary_state);
  }

  auto combine(icts::PatternId upstream, icts::PatternId downstream) const -> icts::PatternId
  {
    LOG_FATAL_IF(!canCompose(upstream, downstream)) << "CharacterizationRealTechFixture: invalid non-monotonic topology composition.";
    const auto upstream_state = LookupTopologyState(*_context, upstream);
    const auto downstream_state = LookupTopologyState(*_context, downstream);

    const auto merged_pattern_id = icts::PatternId::topology(_context->next_pattern_id++);
    _context->composition_states[merged_pattern_id] = icts::PatternCompositionState{
        .terminal_semantic = downstream_state.terminal_semantic,
        .monotonic_boundary_state
        = icts::MonotonicBoundaryState::compose(upstream_state.monotonic_boundary_state, downstream_state.monotonic_boundary_state),
    };
    return merged_pattern_id;
  }

 private:
  HTreeFrontierContext* _context = nullptr;
};

auto MakeSegmentCharTable(const std::vector<icts::SegmentChar>& chars) -> icts::SegmentCharTable
{
  icts::SegmentCharTable table;
  table.reserve(chars.size());
  for (const auto& entry : chars) {
    table.addChar(entry);
  }
  return table;
}

auto MakeHTreeCharTable(const std::vector<icts::HTreeTopologyChar>& chars) -> icts::HTreeTopologyCharTable
{
  icts::HTreeTopologyCharTable table;
  table.reserve(chars.size());
  for (const auto& entry : chars) {
    table.addChar(entry);
  }
  return table;
}

}  // namespace

auto BuildSegmentFrontierContext(const std::vector<icts::BufferingPattern>& patterns) -> SegmentFrontierContext
{
  SegmentFrontierContext context;
  BufferStrengthTable strength_table;
  context.patterns.reserve(patterns.size());
  context.composition_states.reserve(patterns.size());
  for (const auto& pattern : patterns) {
    const auto enriched_pattern = EnrichPatternBoundaryState(pattern, strength_table);
    const auto pattern_id = enriched_pattern.get_pattern_id();
    context.next_pattern_id = std::max(context.next_pattern_id, pattern_id.local_id + 1U);
    context.patterns[pattern_id] = enriched_pattern;
    context.composition_states[pattern_id] = BuildCompositionState(enriched_pattern);
  }
  return context;
}

auto BuildSegmentStateFrontier(const std::vector<icts::SegmentChar>& chars, const SegmentFrontierContext& context)
    -> std::vector<icts::SegmentChar>
{
  return icts::BuildSegmentStateFrontier(chars, [&](const icts::SegmentChar& entry) -> icts::PatternCompositionState {
    return LookupSegmentState(context, entry.get_pattern_id());
  });
}

auto BuildHTreeStateFrontier(const std::vector<icts::HTreeTopologyChar>& chars, const HTreeFrontierContext& context)
    -> std::vector<icts::HTreeTopologyChar>
{
  return icts::BuildHTreeStateFrontier(chars, [&](const icts::HTreeTopologyChar& entry) -> icts::PatternCompositionState {
    return LookupTopologyState(context, entry.get_pattern_id());
  });
}

auto MakeLengthIndex(double length_um, double length_step_um) -> unsigned
{
  if (length_step_um <= 0.0) {
    return 0U;
  }
  return icts::UniformValueLattice(length_step_um, std::numeric_limits<unsigned>::max()).coveringIndex(length_um);
}

auto CalcCharGrid(const icts::CharBuilder& builder) -> CharGrid
{
  return CharGrid{
      .length_step_um = builder.get_wirelength_unit_um(),
      .slew_step_ns = builder.get_slew_steps() == 0U ? 0.0 : builder.get_max_slew() / static_cast<double>(builder.get_slew_steps()),
      .cap_step_pf = builder.get_cap_steps() == 0U ? 0.0 : builder.get_max_cap() / static_cast<double>(builder.get_cap_steps()),
  };
}

auto SummarizeSegmentCharLattice(const std::vector<icts::SegmentChar>& chars, const icts::CharBuilder& builder) -> SegmentCharLatticeSummary
{
  SegmentCharLatticeSummary summary;
  summary.total_entries = chars.size();

  for (const auto& entry : chars) {
    summary.max_length_idx = std::max(summary.max_length_idx, entry.get_length_idx());
    summary.max_input_slew_idx = std::max(summary.max_input_slew_idx, entry.get_input_slew_idx());
    summary.max_output_slew_idx = std::max(summary.max_output_slew_idx, entry.get_output_slew_idx());
    summary.max_driven_cap_idx = std::max(summary.max_driven_cap_idx, entry.get_driven_cap_idx());
    summary.max_load_cap_idx = std::max(summary.max_load_cap_idx, entry.get_load_cap_idx());

    bool entry_out_of_range = false;
    if (entry.get_length_idx() > builder.get_wirelength_iterations()) {
      ++summary.length_overflow_entries;
      entry_out_of_range = true;
    }
    if (entry.get_input_slew_idx() > builder.get_slew_steps()) {
      ++summary.input_slew_overflow_entries;
      entry_out_of_range = true;
    }
    if (entry.get_output_slew_idx() > builder.get_slew_steps()) {
      ++summary.output_slew_overflow_entries;
      entry_out_of_range = true;
    }
    if (entry.get_driven_cap_idx() > builder.get_cap_steps()) {
      ++summary.driven_cap_overflow_entries;
      entry_out_of_range = true;
    }
    if (entry.get_load_cap_idx() > builder.get_cap_steps()) {
      ++summary.load_cap_overflow_entries;
      entry_out_of_range = true;
    }
    if (entry_out_of_range) {
      ++summary.out_of_range_entries;
    }
  }

  return summary;
}

auto FormatSegmentCharLatticeSummary(const SegmentCharLatticeSummary& summary, const icts::CharBuilder& builder) -> std::string
{
  std::ostringstream output_stream;
  output_stream << "total=" << summary.total_entries << ", out_of_range=" << summary.out_of_range_entries
                << ", max_length_idx=" << summary.max_length_idx << "/" << builder.get_wirelength_iterations()
                << ", max_input_slew_idx=" << summary.max_input_slew_idx << "/" << builder.get_slew_steps()
                << ", max_output_slew_idx=" << summary.max_output_slew_idx << "/" << builder.get_slew_steps()
                << ", max_driven_cap_idx=" << summary.max_driven_cap_idx << "/" << builder.get_cap_steps()
                << ", max_load_cap_idx=" << summary.max_load_cap_idx << "/" << builder.get_cap_steps()
                << ", field_overflows{length=" << summary.length_overflow_entries << ",input_slew=" << summary.input_slew_overflow_entries
                << ",output_slew=" << summary.output_slew_overflow_entries << ",driven_cap=" << summary.driven_cap_overflow_entries
                << ",load_cap=" << summary.load_cap_overflow_entries << "}";
  return output_stream.str();
}

auto ComposeSegmentEntriesExact(const std::vector<icts::SegmentChar>& upstream, const std::vector<icts::SegmentChar>& downstream,
                                SegmentFrontierContext& context) -> std::vector<icts::SegmentChar>
{
  SegmentStateCombiner combiner(&context);
  auto composed_entries = MakeSegmentCharTable(upstream).concatWith(MakeSegmentCharTable(downstream), combiner).get_chars();
  SortCharsForReport(composed_entries);
  return composed_entries;
}

auto BuildSegmentLengthFrontiers(const std::vector<icts::SegmentChar>& chars, const SegmentFrontierContext& context)
    -> std::unordered_map<unsigned, std::vector<icts::SegmentChar>>
{
  std::unordered_map<unsigned, std::vector<icts::SegmentChar>> raw_by_length;
  raw_by_length.reserve(chars.size());
  for (const auto& entry : chars) {
    raw_by_length[entry.get_length_idx()].push_back(entry);
  }

  std::unordered_map<unsigned, std::vector<icts::SegmentChar>> frontier_by_length;
  frontier_by_length.reserve(raw_by_length.size());
  for (auto& [length_idx, raw_entries] : raw_by_length) {
    (void) length_idx;
    frontier_by_length[length_idx] = BuildSegmentStateFrontier(raw_entries, context);
  }

  return frontier_by_length;
}

auto SynthesizeSegmentFrontierExactOnly(std::unordered_map<unsigned, std::vector<icts::SegmentChar>>& frontier_by_length,
                                        unsigned target_length_idx, SegmentFrontierContext& context) -> bool
{
  for (unsigned length_idx = 1U; length_idx <= target_length_idx; ++length_idx) {
    if (frontier_by_length.contains(length_idx) && !frontier_by_length.at(length_idx).empty()) {
      continue;
    }

    std::vector<icts::SegmentChar> exact_composed_entries;
    for (unsigned left_idx = 1U; left_idx < length_idx; ++left_idx) {
      const unsigned right_idx = length_idx - left_idx;
      const auto left_it = frontier_by_length.find(left_idx);
      const auto right_it = frontier_by_length.find(right_idx);
      if (left_it == frontier_by_length.end() || right_it == frontier_by_length.end()) {
        continue;
      }
      if (left_it->second.empty() || right_it->second.empty()) {
        continue;
      }

      auto partial = ComposeSegmentEntriesExact(left_it->second, right_it->second, context);
      exact_composed_entries.insert(exact_composed_entries.end(), partial.begin(), partial.end());
    }

    if (!exact_composed_entries.empty()) {
      frontier_by_length[length_idx] = BuildSegmentStateFrontier(exact_composed_entries, context);
    }
  }

  return frontier_by_length.contains(target_length_idx) && !frontier_by_length.at(target_length_idx).empty();
}

auto MakeHTreeSeedEntries(const std::vector<icts::SegmentChar>& segment_frontier, const SegmentFrontierContext& segment_context,
                          HTreeFrontierContext& htree_context) -> std::vector<icts::HTreeTopologyChar>
{
  std::vector<icts::HTreeTopologyChar> seed_entries;
  seed_entries.reserve(segment_frontier.size());
  for (const auto& segment_entry : segment_frontier) {
    const auto topology_pattern_id = icts::PatternId::topology(htree_context.next_pattern_id++);
    htree_context.composition_states[topology_pattern_id] = LookupSegmentState(segment_context, segment_entry.get_pattern_id());
    const icts::CharCore core(segment_entry.get_input_slew_idx(), segment_entry.get_output_slew_idx(), segment_entry.get_driven_cap_idx(),
                              segment_entry.get_load_cap_idx(), segment_entry.get_delay(), segment_entry.get_power(), topology_pattern_id,
                              segment_entry.get_source_boundary_net_switch_power());
    seed_entries.emplace_back(core, 1U);
  }
  SortCharsForReport(seed_entries);
  return seed_entries;
}

auto ComposeHTreeEntriesExact(const std::vector<icts::HTreeTopologyChar>& upstream, const std::vector<icts::HTreeTopologyChar>& downstream,
                              HTreeFrontierContext& htree_context) -> std::vector<icts::HTreeTopologyChar>
{
  TopologyStateCombiner combiner(&htree_context);
  auto composed_entries = MakeHTreeCharTable(upstream).concatWith(MakeHTreeCharTable(downstream), combiner).get_chars();
  SortCharsForReport(composed_entries);
  return composed_entries;
}

auto CountPositivePower(const std::vector<icts::SegmentChar>& chars) -> std::size_t
{
  return static_cast<std::size_t>(
      std::ranges::count_if(chars, [](const icts::SegmentChar& entry) -> bool { return entry.get_power() > 0.0; }));
}

auto FormatSegmentChar(const icts::SegmentChar& entry, const CharGrid& grid) -> std::string
{
  std::ostringstream output_stream;
  output_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  output_stream << std::setprecision(3) << "{length_um=" << (entry.get_length_idx() * grid.length_step_um)
                << ", input_slew_ns=" << (entry.get_input_slew_idx() * grid.slew_step_ns)
                << ", output_slew_ns=" << (entry.get_output_slew_idx() * grid.slew_step_ns)
                << ", driven_cap_pf=" << (entry.get_driven_cap_idx() * grid.cap_step_pf)
                << ", load_cap_pf=" << (entry.get_load_cap_idx() * grid.cap_step_pf) << ", delay_ns=" << entry.get_delay();
  output_stream << ", power_w=" << std::scientific << std::setprecision(6) << entry.get_power() << "}";
  return output_stream.str();
}

auto FormatHTreeChar(const icts::HTreeTopologyChar& entry, const CharGrid& grid) -> std::string
{
  std::ostringstream output_stream;
  output_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  output_stream << std::setprecision(3) << "{levels=" << entry.get_levels()
                << ", input_slew_ns=" << (entry.get_input_slew_idx() * grid.slew_step_ns)
                << ", output_slew_ns=" << (entry.get_output_slew_idx() * grid.slew_step_ns)
                << ", driven_cap_pf=" << (entry.get_driven_cap_idx() * grid.cap_step_pf)
                << ", load_cap_pf=" << (entry.get_load_cap_idx() * grid.cap_step_pf) << ", delay_ns=" << entry.get_delay();
  output_stream << ", power_w=" << std::scientific << std::setprecision(6) << entry.get_power() << "}";
  return output_stream.str();
}

auto SelectBestHTreeChar(const std::vector<icts::HTreeTopologyChar>& entries) -> std::optional<icts::HTreeTopologyChar>
{
  if (entries.empty()) {
    return std::nullopt;
  }

  auto best = entries.front();
  for (const auto& entry : entries) {
    const bool better = (entry.get_load_cap_idx() > best.get_load_cap_idx())
                        || (entry.get_load_cap_idx() == best.get_load_cap_idx() && entry.get_delay() < best.get_delay())
                        || (entry.get_load_cap_idx() == best.get_load_cap_idx() && entry.get_delay() == best.get_delay()
                            && entry.get_output_slew_idx() < best.get_output_slew_idx())
                        || (entry.get_load_cap_idx() == best.get_load_cap_idx() && entry.get_delay() == best.get_delay()
                            && entry.get_output_slew_idx() == best.get_output_slew_idx() && entry.get_power() < best.get_power());
    if (better) {
      best = entry;
    }
  }
  return best;
}

}  // namespace icts_test::characterization::realtech
