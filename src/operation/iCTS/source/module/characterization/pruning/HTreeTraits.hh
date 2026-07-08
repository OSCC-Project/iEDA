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
 * @file HTreeTraits.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Traits for H-tree Hash-Join operations.
 */

#pragma once
// IWYU pragma: private, include "characterization/pruning/HTreeTraits.hh"

#include "HTreeTopologyChar.hh"
#include "ValueLattice.hh"
#include "characterization/pruning/HashJoinEngine.hh"

namespace icts {

/**
 * @brief Traits for HTreeTopologyChar Hash-Join operations.
 *
 * Join condition (with deterministic transform):
 * - upstream.output_slew == downstream.input_slew
 * - halfCapKey(upstream.load_cap_idx) == downstream.driven_cap_idx
 *
 * The half-cap transform accounts for binary fan-out.
 */
struct HTreeTraits
{
  static auto halfCapKey(unsigned load_cap_idx) -> unsigned { return CeilDivUnsigned(load_cap_idx, 2U); }

  /**
   * @brief Build key from downstream entry.
   *
   * Key = pack(input_slew, driven_cap)
   */
  static auto buildKey(const HTreeTopologyChar& c) -> unsigned { return detail::Pack(c.get_input_slew_idx(), c.get_driven_cap_idx()); }

  /**
   * @brief Probe key from upstream entry.
   *
   * Key = pack(output_slew_idx, half-load cap key)
   */
  static auto probeKey(const HTreeTopologyChar& c) -> unsigned
  {
    const unsigned half_cap_key = halfCapKey(c.get_load_cap_idx());
    return detail::Pack(c.get_output_slew_idx(), half_cap_key);
  }

  /**
   * @brief Compose upstream and downstream into merged result.
   */
  static auto compose(const HTreeTopologyChar& up, const HTreeTopologyChar& down, PatternId merged_pid) -> HTreeTopologyChar
  {
    return HTreeTopologyChar::compose(up, down, merged_pid);
  }
};

}  // namespace icts
