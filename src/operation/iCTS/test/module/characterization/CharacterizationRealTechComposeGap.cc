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
 * @file CharacterizationRealTechComposeGap.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Exact compose comparison helpers for real-tech characterization regression tests.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {

auto MakePatternSignature(const realtech_fixture::SegmentFrontierContext& segment_context, icts::PatternId pattern_id) -> std::string
{
  const auto pattern_it = segment_context.patterns.find(pattern_id);
  if (pattern_it == segment_context.patterns.end()) {
    std::ostringstream stream;
    stream << "missing{domain=" << static_cast<unsigned>(pattern_id.domain) << ",local_id=" << pattern_id.local_id << "}";
    return stream.str();
  }

  const auto& pattern = pattern_it->second;
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream << std::setprecision(6);
  stream << "len=" << pattern.get_length_idx();
  stream << ";terminal=" << (pattern.hasTerminalBranchBuffer() ? 1 : 0);
  stream << ";src_has=" << (pattern.get_monotonic_boundary_state().source.has_buffer ? 1 : 0);
  stream << ";src_rank=" << pattern.get_monotonic_boundary_state().source.strength_rank;
  stream << ";snk_has=" << (pattern.get_monotonic_boundary_state().sink.has_buffer ? 1 : 0);
  stream << ";snk_rank=" << pattern.get_monotonic_boundary_state().sink.strength_rank;
  stream << ";pos=";
  for (double pos : pattern.get_buffer_positions()) {
    stream << pos << ",";
  }
  stream << ";masters=";
  for (const auto& cell_master : pattern.get_cell_masters()) {
    stream << cell_master << ",";
  }
  return stream.str();
}

auto MakeSegmentCompareKey(const realtech_fixture::SegmentFrontierContext& segment_context, const icts::SegmentChar& entry)
    -> SegmentCompareKey
{
  return SegmentCompareKey{
      .pattern_signature = MakePatternSignature(segment_context, entry.get_pattern_id()),
      .input_slew_idx = entry.get_input_slew_idx(),
      .output_slew_idx = entry.get_output_slew_idx(),
      .driven_cap_idx = entry.get_driven_cap_idx(),
      .load_cap_idx = entry.get_load_cap_idx(),
  };
}

auto IsPreferredDirectEntry(const icts::SegmentChar& candidate, const icts::SegmentChar& current) -> bool
{
  if (candidate.get_delay() != current.get_delay()) {
    return candidate.get_delay() < current.get_delay();
  }
  return candidate.get_power() < current.get_power();
}

auto SafeRatio(double numerator, double denominator) -> double
{
  return std::abs(denominator) > 1e-30 ? numerator / denominator : 0.0;
}

auto CompareComposedFrontierToDirect(unsigned target_length_idx, const std::vector<icts::SegmentChar>& direct_entries,
                                     const std::vector<icts::SegmentChar>& composed_entries,
                                     const realtech_fixture::SegmentFrontierContext& segment_context,
                                     const realtech_fixture::CharGrid& grid) -> ComposeGapStats
{
  ComposeGapStats stats;
  stats.target_length_idx = target_length_idx;
  stats.direct_count = direct_entries.size();
  stats.composed_count = composed_entries.size();

  std::unordered_map<SegmentCompareKey, icts::SegmentChar, SegmentCompareKeyHash> direct_by_key;
  direct_by_key.reserve(direct_entries.size());
  for (const auto& entry : direct_entries) {
    const auto key = MakeSegmentCompareKey(segment_context, entry);
    auto [it, inserted] = direct_by_key.emplace(key, entry);
    if (!inserted) {
      ++stats.direct_duplicate_key_count;
      if (IsPreferredDirectEntry(entry, it->second)) {
        it->second = entry;
      }
    }
  }

  for (const auto& composed_entry : composed_entries) {
    const auto direct_it = direct_by_key.find(MakeSegmentCompareKey(segment_context, composed_entry));
    if (direct_it == direct_by_key.end()) {
      ++stats.missing_composed_count;
      if (stats.missing_composed_example.empty()) {
        stats.missing_composed_example = realtech_fixture::FormatSegmentChar(composed_entry, grid);
      }
      continue;
    }

    const auto& direct_entry = direct_it->second;
    ++stats.matched_count;
    stats.sum_direct_delay_ns += direct_entry.get_delay();
    stats.sum_composed_delay_ns += composed_entry.get_delay();
    stats.sum_direct_power_w += direct_entry.get_power();
    stats.sum_composed_power_w += composed_entry.get_power();

    const double delay_delta_ns = composed_entry.get_delay() - direct_entry.get_delay();
    const double abs_delay_delta_ns = std::abs(delay_delta_ns);
    stats.sum_abs_delay_delta_ns += abs_delay_delta_ns;
    stats.sum_sq_delay_delta_ns += delay_delta_ns * delay_delta_ns;
    stats.max_rel_delay_delta
        = std::max(stats.max_rel_delay_delta, abs_delay_delta_ns / std::max(std::abs(direct_entry.get_delay()), 1e-30));
    if (delay_delta_ns < -1e-15) {
      ++stats.composed_lower_delay_count;
    } else if (delay_delta_ns > 1e-15) {
      ++stats.composed_higher_delay_count;
    }
    if (abs_delay_delta_ns > stats.max_abs_delay_delta_ns) {
      stats.max_abs_delay_delta_ns = abs_delay_delta_ns;
      std::ostringstream example;
      example << "direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
              << " composed=" << realtech_fixture::FormatSegmentChar(composed_entry, grid);
      stats.worst_delay_example = example.str();
    }

    const double power_delta_w = composed_entry.get_power() - direct_entry.get_power();
    const double abs_power_delta_w = std::abs(power_delta_w);
    stats.sum_abs_power_delta_w += abs_power_delta_w;
    stats.sum_sq_power_delta_w += power_delta_w * power_delta_w;
    stats.max_rel_power_delta
        = std::max(stats.max_rel_power_delta, abs_power_delta_w / std::max(std::abs(direct_entry.get_power()), 1e-30));
    if (power_delta_w < -1e-15) {
      ++stats.composed_lower_power_count;
    } else if (power_delta_w > 1e-15) {
      ++stats.composed_higher_power_count;
    }
    if (abs_power_delta_w > stats.max_abs_power_delta_w) {
      stats.max_abs_power_delta_w = abs_power_delta_w;
      std::ostringstream example;
      example << "direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
              << " composed=" << realtech_fixture::FormatSegmentChar(composed_entry, grid);
      stats.worst_power_example = example.str();
    }
  }

  return stats;
}

auto AppendComposeGapStats(std::ostringstream& report_stream, const ComposeGapStats& stats, const realtech_fixture::CharGrid& grid) -> void
{
  const auto matched_count = static_cast<double>(stats.matched_count);
  const double delay_rmse_ns = stats.matched_count == 0U ? 0.0 : std::sqrt(stats.sum_sq_delay_delta_ns / matched_count);
  const double delay_mean_abs_ns = stats.matched_count == 0U ? 0.0 : stats.sum_abs_delay_delta_ns / matched_count;
  const double power_rmse_w = stats.matched_count == 0U ? 0.0 : std::sqrt(stats.sum_sq_power_delta_w / matched_count);
  const double power_mean_abs_w = stats.matched_count == 0U ? 0.0 : stats.sum_abs_power_delta_w / matched_count;

  report_stream << "compose_gap{target_length_idx=" << stats.target_length_idx
                << ",target_length_um=" << (static_cast<double>(stats.target_length_idx) * grid.length_step_um)
                << ",direct_frontier_count=" << stats.direct_count << ",composed_frontier_count=" << stats.composed_count
                << ",direct_duplicate_key_count=" << stats.direct_duplicate_key_count << ",matched_count=" << stats.matched_count
                << ",missing_composed_count=" << stats.missing_composed_count
                << ",match_over_composed=" << SafeRatio(static_cast<double>(stats.matched_count), static_cast<double>(stats.composed_count))
                << ",match_over_direct=" << SafeRatio(static_cast<double>(stats.matched_count), static_cast<double>(stats.direct_count))
                << ",sum_direct_delay_ns=" << stats.sum_direct_delay_ns << ",sum_composed_delay_ns=" << stats.sum_composed_delay_ns
                << ",delay_sum_delta_ns=" << (stats.sum_composed_delay_ns - stats.sum_direct_delay_ns)
                << ",delay_ratio_composed_over_direct=" << SafeRatio(stats.sum_composed_delay_ns, stats.sum_direct_delay_ns)
                << ",delay_rmse_ns=" << delay_rmse_ns << ",delay_mean_abs_ns=" << delay_mean_abs_ns
                << ",delay_max_abs_ns=" << stats.max_abs_delay_delta_ns << ",delay_max_rel=" << stats.max_rel_delay_delta
                << ",composed_lower_delay_count=" << stats.composed_lower_delay_count
                << ",composed_higher_delay_count=" << stats.composed_higher_delay_count
                << ",sum_direct_power_w=" << stats.sum_direct_power_w << ",sum_composed_power_w=" << stats.sum_composed_power_w
                << ",power_sum_delta_w=" << (stats.sum_composed_power_w - stats.sum_direct_power_w)
                << ",power_ratio_composed_over_direct=" << SafeRatio(stats.sum_composed_power_w, stats.sum_direct_power_w)
                << ",power_rmse_w=" << power_rmse_w << ",power_mean_abs_w=" << power_mean_abs_w
                << ",power_max_abs_w=" << stats.max_abs_power_delta_w << ",power_max_rel=" << stats.max_rel_power_delta
                << ",composed_lower_power_count=" << stats.composed_lower_power_count
                << ",composed_higher_power_count=" << stats.composed_higher_power_count << "}\n";
  report_stream << "compose_gap_worst_delay{target_length_idx=" << stats.target_length_idx << "," << stats.worst_delay_example << "}\n";
  report_stream << "compose_gap_worst_power{target_length_idx=" << stats.target_length_idx << "," << stats.worst_power_example << "}\n";
  report_stream << "compose_gap_missing_example{target_length_idx=" << stats.target_length_idx << "," << stats.missing_composed_example
                << "}\n";
}

}  // namespace icts_test
