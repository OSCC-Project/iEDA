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
 * @file HashJoinEngine.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-30
 * @brief Header-only Hash-Join engine for table concatenation.
 *
 * Implements build+probe Hash-Join with compile-time traits injection.
 * Zero virtual function overhead - all polymorphism resolved at compile time.
 */

#pragma once
// IWYU pragma: private, include "characterization/pruning/HashJoinEngine.hh"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace icts {
namespace detail {

/**
 * @brief Pack slew and cap into a single 32-bit key.
 *
 * Key format: [slew:16][cap:16]
 */
inline auto Pack(unsigned slew, unsigned cap) -> unsigned
{
  return (slew << 16) | cap;
}

/**
 * @brief Null pruner type for when pruning is disabled.
 */
struct NullPruner
{
};

template <class PrunerT>
using PrunerGroupKeyT = typename PrunerT::GroupKey;

template <class PrunerT>
using PrunerGroupKeyHashT = typename PrunerT::GroupKeyHash;

template <class CombinerT, class CharT>
inline auto CanCompose(const CombinerT& combiner, const CharT& upstream, const CharT& downstream) -> bool
{
  if constexpr (requires { combiner.canCompose(upstream.get_pattern_id(), downstream.get_pattern_id()); }) {
    return static_cast<bool>(combiner.canCompose(upstream.get_pattern_id(), downstream.get_pattern_id()));
  }
  return true;
}

/**
 * @brief Hash-Join concatenation of two tables.
 *
 * Performs Hash-Join with:
 * - Build phase: index downstream by buildKey
 * - Probe phase: probe upstream against index using probeKey
 *
 * Complexity:
 * - Average: O(|U| + |D| + J) where J = number of matches
 * - Worst: O(|U| × |D|) if all keys collide
 *
 * @tparam CharT Characterization type (SegmentChar or HTreeTopologyChar)
 * @tparam Traits Traits class providing buildKey, probeKey, compose
 * @tparam CombinerT Pattern combiner type
 * @tparam PrunerT Pruner type (use NullPruner to disable)
 *
 * @param upstream Upstream characterization entries
 * @param downstream Downstream characterization entries
 * @param combiner Pattern combiner for creating merged pattern IDs
 * @param out Output vector for composed results
 * @param pruner Optional pruner (nullptr to disable)
 */
template <class CharT, class Traits, class CombinerT, class PrunerT>
inline auto HashJoinConcat(const std::vector<CharT>& upstream, const std::vector<CharT>& downstream, const CombinerT& combiner,
                           std::vector<CharT>& out, [[maybe_unused]] const PrunerT* pruner = nullptr) -> void
{
  if (upstream.empty() || downstream.empty()) {
    return;
  }

  if constexpr (!std::is_same_v<PrunerT, NullPruner>) {
    if (pruner != nullptr) {
      std::unordered_map<PrunerGroupKeyT<PrunerT>, std::size_t, PrunerGroupKeyHashT<PrunerT>> group_to_index;
      std::vector<std::vector<CharT>> grouped_out;
      grouped_out.reserve(out.size());

      for (auto& existing : out) {
        const auto group = pruner->groupKey(existing);
        auto [it, inserted] = group_to_index.emplace(group, grouped_out.size());
        if (inserted) {
          grouped_out.emplace_back();
        }
        grouped_out[it->second].push_back(std::move(existing));
      }
      out.clear();

      // Build phase: hash downstream entries by buildKey
      std::unordered_map<unsigned, std::vector<std::size_t>> index;
      index.reserve(downstream.size());
      for (std::size_t i = 0; i < downstream.size(); ++i) {
        index[Traits::buildKey(downstream[i])].push_back(i);
      }

      for (const auto& up : upstream) {
        unsigned key = Traits::probeKey(up);
        auto it = index.find(key);
        if (it == index.end()) {
          continue;
        }

        for (std::size_t di : it->second) {
          const auto& down = downstream[di];
          if (!CanCompose(combiner, up, down)) {
            continue;
          }
          auto merged_pid = combiner.combine(up.get_pattern_id(), down.get_pattern_id());
          CharT result = Traits::compose(up, down, merged_pid);
          const auto group = pruner->groupKey(result);

          auto [group_it, inserted] = group_to_index.emplace(group, grouped_out.size());
          if (inserted) {
            grouped_out.emplace_back();
          }
          auto& frontier = grouped_out[group_it->second];

          bool dominated = false;
          for (const auto& existing : frontier) {
            if (pruner->dominates(existing, result)) {
              dominated = true;
              break;
            }
          }
          if (dominated) {
            continue;
          }

          auto new_end = std::remove_if(frontier.begin(), frontier.end(),
                                        [&](const CharT& existing) -> bool { return pruner->dominates(result, existing); });
          frontier.erase(new_end, frontier.end());
          frontier.push_back(std::move(result));
        }
      }

      std::size_t total_size = 0;
      for (const auto& group_entries : grouped_out) {
        total_size += group_entries.size();
      }
      out.reserve(total_size);
      for (auto& group_entries : grouped_out) {
        for (auto& entry : group_entries) {
          out.push_back(std::move(entry));
        }
      }
      return;
    }
  }

  // Build phase: hash downstream entries by buildKey
  std::unordered_map<unsigned, std::vector<std::size_t>> index;
  index.reserve(downstream.size());
  for (std::size_t i = 0; i < downstream.size(); ++i) {
    index[Traits::buildKey(downstream[i])].push_back(i);
  }

  out.reserve(out.size() + upstream.size());
  for (const auto& up : upstream) {
    unsigned key = Traits::probeKey(up);
    auto it = index.find(key);
    if (it == index.end()) {
      continue;
    }

    for (std::size_t di : it->second) {
      const auto& down = downstream[di];
      if (!CanCompose(combiner, up, down)) {
        continue;
      }
      auto merged_pid = combiner.combine(up.get_pattern_id(), down.get_pattern_id());
      out.push_back(Traits::compose(up, down, merged_pid));
    }
  }
}

}  // namespace detail
}  // namespace icts
