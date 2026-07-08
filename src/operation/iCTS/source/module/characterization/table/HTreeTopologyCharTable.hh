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
 * @file HTreeTopologyCharTable.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief H-tree topology characterization table with Hash-Join concatenation.
 */

#pragma once
// IWYU pragma: private, include "characterization/table/HTreeTopologyCharTable.hh"

#include <vector>

#include "HTreeTopologyChar.hh"
#include "characterization/pruning/HTreeTraits.hh"
#include "characterization/pruning/HashJoinEngine.hh"

namespace icts {

/**
 * @brief Table of H-tree topology characterizations.
 *
 * Supports Hash-Join based concatenation with half-cap key mapping.
 */
class HTreeTopologyCharTable
{
 public:
  HTreeTopologyCharTable() = default;

  /**
   * @brief Add a characterization entry.
   */
  auto addChar(HTreeTopologyChar c) -> void { _chars.push_back(std::move(c)); }

  /**
   * @brief Clear all entries.
   */
  auto clear() -> void { _chars.clear(); }

  /**
   * @brief Get read-only access to characterization entries.
   */
  auto get_chars() const -> const std::vector<HTreeTopologyChar>& { return _chars; }

  /**
   * @brief Get the number of entries.
   */
  auto size() const -> std::size_t { return _chars.size(); }

  /**
   * @brief Reserve capacity.
   */
  auto reserve(std::size_t n) -> void { _chars.reserve(n); }

  /**
   * @brief Concatenate with downstream table using Hash-Join.
   *
   * Join condition (with transform):
   * - this.output_slew == downstream.input_slew
   * - halfCapKey(this.load_cap_idx) == downstream.driven_cap_idx
   *
   * @tparam CombinerT Pattern combiner type
   * @tparam PrunerT Pruner type (default: NullPruner for no pruning)
   * @param downstream Downstream table to join with
   * @param combiner Pattern combiner for merged pattern IDs
   * @param pruner Optional pruner (nullptr to disable)
   * @return New table with composed entries
   */
  template <class CombinerT, class PrunerT = detail::NullPruner>
  auto concatWith(const HTreeTopologyCharTable& downstream, const CombinerT& combiner, const PrunerT* pruner = nullptr) const
      -> HTreeTopologyCharTable
  {
    HTreeTopologyCharTable result;
    detail::HashJoinConcat<HTreeTopologyChar, HTreeTraits>(_chars, downstream._chars, combiner, result._chars, pruner);
    return result;
  }

 private:
  std::vector<HTreeTopologyChar> _chars;
};

}  // namespace icts
