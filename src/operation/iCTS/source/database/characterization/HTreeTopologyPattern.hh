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
 * @file HTreeTopologyPattern.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief H-tree topology pattern storing level segment references.
 */

#pragma once

#include <vector>

#include "PatternId.hh"

namespace icts {

/**
 * @brief H-tree topology pattern.
 *
 * Represents a symmetric H-tree structure by storing only level count and
 * references to segment patterns at each level. Does NOT store actual tree
 * topology to avoid memory explosion - structure is derivable from the
 * symmetric property.
 */
class HTreeTopologyPattern
{
 public:
  HTreeTopologyPattern() = default;

  HTreeTopologyPattern(PatternId pattern_id, unsigned levels, std::vector<PatternId> level_segment_ids)
      : _pattern_id(pattern_id), _levels(levels), _level_segment_pattern_ids(std::move(level_segment_ids))
  {
  }

  // Getters
  auto get_pattern_id() const -> PatternId { return _pattern_id; }
  auto get_levels() const -> unsigned { return _levels; }
  auto get_level_segment_pattern_ids() const -> const std::vector<PatternId>& { return _level_segment_pattern_ids; }

  /**
   * @brief Concatenate two H-tree topology patterns.
   *
   * Combines levels and segment pattern references.
   * The actual topology is not expanded - only metadata is merged.
   *
   * @param upstream Upstream topology (closer to root)
   * @param downstream Downstream topology (closer to leaves)
   * @param merged_topo_pid New pattern ID for the merged topology
   */
  static auto concat(const HTreeTopologyPattern& upstream, const HTreeTopologyPattern& downstream, PatternId merged_topo_pid)
      -> HTreeTopologyPattern
  {
    std::vector<PatternId> merged_ids;
    merged_ids.reserve(upstream._level_segment_pattern_ids.size() + downstream._level_segment_pattern_ids.size());
    merged_ids.insert(merged_ids.end(), upstream._level_segment_pattern_ids.begin(), upstream._level_segment_pattern_ids.end());
    merged_ids.insert(merged_ids.end(), downstream._level_segment_pattern_ids.begin(), downstream._level_segment_pattern_ids.end());

    return HTreeTopologyPattern(merged_topo_pid, upstream._levels + downstream._levels, std::move(merged_ids));
  }

 private:
  PatternId _pattern_id{PatternDomain::kTopologyPattern, 0};
  unsigned _levels = 0;
  std::vector<PatternId> _level_segment_pattern_ids;  ///< domain=kSegmentPattern
};

}  // namespace icts
