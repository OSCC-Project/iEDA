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
 * @file AnalyticalCandidate.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Mathematical analytical H-tree candidate data types.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "PatternId.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"

namespace icts {
class UniformValueLattice;
}  // namespace icts

namespace icts::htree {
struct BufferPatternLibrary;
}  // namespace icts::htree

namespace icts::htree::analytical_solver {

struct AnalyticalSegmentChoice
{
  std::size_t level_index = 0U;
  unsigned length_idx = 0U;
  PatternId segment_pattern_id = PatternId::segment(0U);
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
};

struct AnalyticalCandidate
{
  unsigned depth = 0U;
  std::size_t leaf_count = 0U;
  double leaf_load_cap_pf = 0.0;
  double root_input_slew_ns = 0.0;
  double root_source_cap_pf = 0.0;
  double output_slew_ns = 0.0;
  double raw_delay_ns = 0.0;
  double raw_power_w = 0.0;
  double conservative_slew_ns = 0.0;
  double conservative_delay_ns = 0.0;
  double conservative_power_w = 0.0;
  bool branch_buffer_legal = true;
  bool fanout_legal = true;
  std::string rejection_reason;
  std::vector<PatternId> level_segment_pattern_ids;
  std::vector<AnalyticalSegmentChoice> trace;
  std::optional<HTreeTopologyChar> materialized_char = std::nullopt;
  TopologyPatternLibrary topology_pattern_library;

  auto isValid() const -> bool;
};

auto PreferAnalyticalCandidate(const AnalyticalCandidate& lhs, const AnalyticalCandidate& rhs) -> bool;
auto LexicographicalPatternIdLess(const std::vector<PatternId>& lhs, const std::vector<PatternId>& rhs) -> bool;
auto BuildAnalyticalTopologyPattern(const std::vector<PatternId>& level_segment_pattern_ids,
                                    const BufferPatternLibrary& segment_pattern_library, std::size_t max_fanout)
    -> std::optional<TopologyPatternLibrary>;
auto MaterializeAnalyticalTopologyChar(const AnalyticalCandidate& candidate, const icts::UniformValueLattice& slew_lattice,
                                       const icts::UniformValueLattice& cap_lattice) -> std::optional<HTreeTopologyChar>;

}  // namespace icts::htree::analytical_solver
