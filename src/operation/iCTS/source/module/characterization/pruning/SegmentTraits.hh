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
 * @file SegmentTraits.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Traits for segment Hash-Join operations.
 */

#pragma once
// IWYU pragma: private, include "characterization/pruning/SegmentTraits.hh"

#include "SegmentChar.hh"
#include "characterization/pruning/HashJoinEngine.hh"

namespace icts {

/**
 * @brief Traits for SegmentChar Hash-Join operations.
 *
 * Join condition (equal):
 * - upstream.output_slew == downstream.input_slew
 * - upstream.load_cap == downstream.driven_cap
 */
struct SegmentTraits
{
  /**
   * @brief Build key from downstream entry.
   *
   * Key = pack(input_slew, driven_cap)
   */
  static auto buildKey(const SegmentChar& c) -> unsigned { return detail::Pack(c.get_input_slew_idx(), c.get_driven_cap_idx()); }

  /**
   * @brief Probe key from upstream entry.
   *
   * Key = pack(output_slew_idx, load_cap_idx)
   */
  static auto probeKey(const SegmentChar& c) -> unsigned { return detail::Pack(c.get_output_slew_idx(), c.get_load_cap_idx()); }

  /**
   * @brief Compose upstream and downstream into merged result.
   */
  static auto compose(const SegmentChar& up, const SegmentChar& down, PatternId merged_pid) -> SegmentChar
  {
    return SegmentChar::compose(up, down, merged_pid);
  }
};

}  // namespace icts
