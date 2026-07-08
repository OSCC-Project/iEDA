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
 * @file SegmentCharTable.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Segment characterization table with Hash-Join concatenation.
 */

#pragma once
// IWYU pragma: private, include "characterization/table/SegmentCharTable.hh"

#include <vector>

#include "SegmentChar.hh"
#include "characterization/pruning/HashJoinEngine.hh"
#include "characterization/pruning/SegmentTraits.hh"

namespace icts {

/**
 * @brief Table of segment characterizations.
 *
 * Supports Hash-Join based concatenation for efficient table composition.
 */
class SegmentCharTable
{
 public:
  SegmentCharTable() = default;

  /**
   * @brief Add a characterization entry.
   */
  auto addChar(SegmentChar c) -> void { _chars.push_back(std::move(c)); }

  /**
   * @brief Clear all entries.
   */
  auto clear() -> void { _chars.clear(); }

  /**
   * @brief Get read-only access to characterization entries.
   */
  auto get_chars() const -> const std::vector<SegmentChar>& { return _chars; }

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
   * Join condition (equal):
   * - this.output_slew == downstream.input_slew
   * - this.load_cap == downstream.driven_cap
   *
   * @tparam CombinerT Pattern combiner type
   * @tparam PrunerT Pruner type (default: NullPruner for no pruning)
   * @param downstream Downstream table to join with
   * @param combiner Pattern combiner for merged pattern IDs
   * @param pruner Optional pruner (nullptr to disable)
   * @return New table with composed entries
   */
  template <class CombinerT, class PrunerT = detail::NullPruner>
  auto concatWith(const SegmentCharTable& downstream, const CombinerT& combiner, const PrunerT* pruner = nullptr) const -> SegmentCharTable
  {
    SegmentCharTable result;
    detail::HashJoinConcat<SegmentChar, SegmentTraits>(_chars, downstream._chars, combiner, result._chars, pruner);
    return result;
  }

 private:
  std::vector<SegmentChar> _chars;
};

}  // namespace icts
