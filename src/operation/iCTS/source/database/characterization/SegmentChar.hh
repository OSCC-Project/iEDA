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
 * @file SegmentChar.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Segment characterization with length and electrical properties.
 */

#pragma once

#include "CharCore.hh"
#include "PatternId.hh"

namespace icts {

/**
 * @brief Segment characterization entry.
 *
 * Combines CharCore (electrical boundaries + cost) with segment length.
 * Supports composition for segment concatenation via Hash-Join.
 *
 * Discretized index fields (*_idx) are integer bin indices in [1, *_steps].
 * Physical values can be recovered via: value = idx * step_size.
 */
class SegmentChar
{
 public:
  SegmentChar() = default;

  SegmentChar(CharCore core, unsigned length_idx) : _core(std::move(core)), _length_idx(length_idx) {}

  // Forwarded getters from CharCore
  auto get_input_slew_idx() const -> unsigned { return _core.get_input_slew_idx(); }
  auto get_output_slew_idx() const -> unsigned { return _core.get_output_slew_idx(); }
  auto get_driven_cap_idx() const -> unsigned { return _core.get_driven_cap_idx(); }
  auto get_load_cap_idx() const -> unsigned { return _core.get_load_cap_idx(); }
  auto get_delay() const -> double { return _core.get_delay(); }
  auto get_power() const -> double { return _core.get_power(); }
  auto get_pattern_id() const -> PatternId { return _core.get_pattern_id(); }
  auto get_source_boundary_net_switch_power() const -> double { return _core.get_source_boundary_net_switch_power(); }

  // Segment-specific getter
  auto get_length_idx() const -> unsigned { return _length_idx; }

  /**
   * @brief Compose two segment characterizations.
   *
   * Composition rules (fixed):
   * - length_idx = upstream.length_idx + downstream.length_idx
   * - input_slew_idx = upstream.input_slew_idx
   * - output_slew_idx = downstream.output_slew_idx
   * - driven_cap_idx = upstream.driven_cap_idx (unchanged from source)
   * - load_cap_idx = downstream.load_cap_idx
   * - delay = upstream.delay + downstream.delay
   * - power = upstream.power + downstream.power - downstream.source_boundary_net_switch_power
   * - source_boundary_net_switch_power = upstream.source_boundary_net_switch_power
   *
   * @param upstream Upstream segment (closer to source)
   * @param downstream Downstream segment (closer to sink)
   * @param merged_pid Pattern ID for the composed segment
   */
  static auto compose(const SegmentChar& upstream, const SegmentChar& downstream, PatternId merged_pid) -> SegmentChar
  {
    CharCore merged_core(upstream.get_input_slew_idx(),                  // input from upstream
                         downstream.get_output_slew_idx(),               // output from downstream
                         upstream.get_driven_cap_idx(),                  // driven_cap from upstream (source side)
                         downstream.get_load_cap_idx(),                  // load_cap from downstream (sink side)
                         upstream.get_delay() + downstream.get_delay(),  // additive delay
                         upstream.get_power() + downstream.get_power() - downstream.get_source_boundary_net_switch_power(), merged_pid,
                         upstream.get_source_boundary_net_switch_power());
    return SegmentChar(std::move(merged_core), upstream._length_idx + downstream._length_idx);
  }

 private:
  CharCore _core;
  unsigned _length_idx = 0;
};

}  // namespace icts
