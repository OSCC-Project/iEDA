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
 * @file BufferingPattern.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Segment buffering pattern with buffer positions and cell masters.
 */

#pragma once

#include <string>
#include <vector>

#include "PatternId.hh"

namespace icts {

/**
 * @brief One compose-visible boundary buffer signature.
 *
 * `has_buffer=false` represents a pure-wire boundary with no exposed buffer.
 * `has_buffer=true` with `strength_rank>0` represents the monotonic size class
 * used by group/compose semantics. `strength_rank=0` is reserved for an
 * unresolved size class and is kept distinct from pure wire by `has_buffer`.
 */
struct BoundaryBufferState
{
  bool has_buffer = false;
  unsigned strength_rank = 0U;

  auto operator==(const BoundaryBufferState& rhs) const -> bool = default;
};

/**
 * @brief Source/sink boundary state for monotonic composition.
 *
 * Strength ranks are ordered weak -> strong, so a source-to-sink monotonic
 * pattern must keep ranks non-increasing across concatenation boundaries.
 */
struct MonotonicBoundaryState
{
  BoundaryBufferState source{};
  BoundaryBufferState sink{};

  auto operator==(const MonotonicBoundaryState& rhs) const -> bool = default;

  auto canComposeWith(const MonotonicBoundaryState& downstream) const -> bool
  {
    if (!sink.has_buffer || !downstream.source.has_buffer) {
      return true;
    }
    return sink.strength_rank >= downstream.source.strength_rank;
  }

  static auto compose(const MonotonicBoundaryState& upstream, const MonotonicBoundaryState& downstream) -> MonotonicBoundaryState
  {
    return MonotonicBoundaryState{
        .source = upstream.source.has_buffer ? upstream.source : downstream.source,
        .sink = downstream.sink.has_buffer ? downstream.sink : upstream.sink,
    };
  }
};

/**
 * @brief Segment buffering pattern.
 *
 * Represents a wire segment with optional buffer insertions.
 * Buffer positions are normalized to (0, 1] relative to segment length.
 * Supports concatenation for segment composition.
 *
 * _length_idx is a discretized bin index (length_um / length_step_um).
 */
class BufferingPattern
{
 public:
  BufferingPattern() = default;

  BufferingPattern(unsigned length_idx, PatternId pattern_id, std::vector<double> buffer_positions, std::vector<std::string> cell_masters,
                   bool has_terminal_branch_buffer = false, MonotonicBoundaryState monotonic_boundary_state = {})
      : _length_idx(length_idx),
        _pattern_id(pattern_id),
        _buffer_positions(std::move(buffer_positions)),
        _cell_masters(std::move(cell_masters)),
        _has_terminal_branch_buffer(has_terminal_branch_buffer),
        _monotonic_boundary_state(monotonic_boundary_state)
  {
  }

  // Getters
  auto get_length_idx() const -> unsigned { return _length_idx; }
  auto get_pattern_id() const -> PatternId { return _pattern_id; }
  auto get_buffer_positions() const -> const std::vector<double>& { return _buffer_positions; }
  auto get_cell_masters() const -> const std::vector<std::string>& { return _cell_masters; }
  auto hasTerminalBranchBuffer() const -> bool { return _has_terminal_branch_buffer; }
  auto get_monotonic_boundary_state() const -> const MonotonicBoundaryState& { return _monotonic_boundary_state; }

  /**
   * @brief Check if this is a pure wire pattern (no buffers).
   */
  auto isWirePattern() const -> bool { return _buffer_positions.empty(); }

  /**
   * @brief Check if this is a buffer pattern (has buffers).
   */
  auto isBufferPattern() const -> bool { return !_buffer_positions.empty(); }

  /**
   * @brief Concatenate two buffering patterns.
   *
   * Upstream pattern comes first, downstream pattern comes after.
   * Buffer positions are renormalized to the combined length.
   */
  static auto canConcatMonotonic(const BufferingPattern& upstream, const BufferingPattern& downstream) -> bool
  {
    return upstream._monotonic_boundary_state.canComposeWith(downstream._monotonic_boundary_state);
  }

  static auto concat(const BufferingPattern& upstream, const BufferingPattern& downstream) -> BufferingPattern
  {
    unsigned total_length = upstream._length_idx + downstream._length_idx;
    if (total_length == 0) {
      return BufferingPattern{0, PatternId::segment(0), {}, {}, false, {}};
    }

    double up_ratio = static_cast<double>(upstream._length_idx) / total_length;

    // Renormalize upstream positions to [0, up_ratio]
    std::vector<double> merged_positions;
    merged_positions.reserve(upstream._buffer_positions.size() + downstream._buffer_positions.size());
    for (double pos : upstream._buffer_positions) {
      merged_positions.push_back(pos * up_ratio);
    }
    // Offset downstream positions to (up_ratio, 1]
    for (double pos : downstream._buffer_positions) {
      merged_positions.push_back(up_ratio + pos * (1.0 - up_ratio));
    }

    // Concatenate cell masters
    std::vector<std::string> merged_masters;
    merged_masters.reserve(upstream._cell_masters.size() + downstream._cell_masters.size());
    merged_masters.insert(merged_masters.end(), upstream._cell_masters.begin(), upstream._cell_masters.end());
    merged_masters.insert(merged_masters.end(), downstream._cell_masters.begin(), downstream._cell_masters.end());

    // Note: pattern_id for merged pattern should be assigned by the caller
    return BufferingPattern{
        total_length,
        PatternId::segment(0),
        std::move(merged_positions),
        std::move(merged_masters),
        downstream._has_terminal_branch_buffer,
        MonotonicBoundaryState::compose(upstream._monotonic_boundary_state, downstream._monotonic_boundary_state),
    };
  }

 private:
  unsigned _length_idx = 0;
  PatternId _pattern_id{PatternDomain::kSegmentPattern, 0};
  std::vector<double> _buffer_positions;   ///< Normalized positions in (0, 1]
  std::vector<std::string> _cell_masters;  ///< Cell master names for each buffer
  bool _has_terminal_branch_buffer = false;
  MonotonicBoundaryState _monotonic_boundary_state;
};

}  // namespace icts
