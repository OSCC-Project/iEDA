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
 * @file SegmentFrontierCatalog.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief H-tree segment frontier kinds, required length sets, and synthesized frontier catalog.
 */

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SegmentChar.hh"
#include "log/Log.hh"

namespace icts::htree {

enum class SegmentFrontierKind
{
  kAll,
  kTerminalBranchBuffered,
  kTerminalLeafUnbuffered,
};

class SegmentFrontierKindSet
{
 public:
  SegmentFrontierKindSet() = default;

  static auto allOnly() -> SegmentFrontierKindSet
  {
    SegmentFrontierKindSet result;
    result.add(SegmentFrontierKind::kAll);
    return result;
  }

  static auto branchConstrained() -> SegmentFrontierKindSet
  {
    auto result = allOnly();
    result.add(SegmentFrontierKind::kTerminalBranchBuffered);
    return result;
  }

  static auto leafConstrained() -> SegmentFrontierKindSet
  {
    auto result = allOnly();
    result.add(SegmentFrontierKind::kTerminalLeafUnbuffered);
    return result;
  }

  static auto full() -> SegmentFrontierKindSet
  {
    auto result = branchConstrained();
    result.add(SegmentFrontierKind::kTerminalLeafUnbuffered);
    return result;
  }

  auto add(SegmentFrontierKind kind) -> void { _mask |= bit(kind); }
  auto contains(SegmentFrontierKind kind) const -> bool { return (_mask & bit(kind)) != 0U; }
  auto empty() const -> bool { return _mask == 0U; }

  auto normalized() const -> SegmentFrontierKindSet
  {
    auto result = *this;
    if (!result.empty()) {
      result.add(SegmentFrontierKind::kAll);
    }
    return result;
  }

 private:
  static auto bit(SegmentFrontierKind kind) -> unsigned
  {
    switch (kind) {
      case SegmentFrontierKind::kAll:
        return 1U << 0U;
      case SegmentFrontierKind::kTerminalBranchBuffered:
        return 1U << 1U;
      case SegmentFrontierKind::kTerminalLeafUnbuffered:
        return 1U << 2U;
    }
    return 0U;
  }

  unsigned _mask = 0U;
};

struct RequiredSegmentFrontiers
{
  std::vector<unsigned> required_length_indices;
  SegmentFrontierKindSet required_kinds = SegmentFrontierKindSet::full();
};

class SegmentCandidateFrontierSet
{
 public:
  auto hasKind(SegmentFrontierKind kind) const -> bool { return _built_kinds.contains(kind); }

  auto find(SegmentFrontierKind kind) const -> const std::vector<SegmentChar>*
  {
    if (!hasKind(kind)) {
      return nullptr;
    }
    return &entries(kind);
  }

  auto require(SegmentFrontierKind kind) const -> const std::vector<SegmentChar>&
  {
    const auto* frontier = find(kind);
    LOG_FATAL_IF(frontier == nullptr) << "HTree: required segment frontier kind was not synthesized.";
    return *frontier;
  }

  auto mutableEntries(SegmentFrontierKind kind) -> std::vector<SegmentChar>&
  {
    _built_kinds.add(kind);
    switch (kind) {
      case SegmentFrontierKind::kAll:
        return _all_frontier_entries;
      case SegmentFrontierKind::kTerminalBranchBuffered:
        return _branch_buffered_entries;
      case SegmentFrontierKind::kTerminalLeafUnbuffered:
        return _leaf_unbuffered_entries;
    }
    return _all_frontier_entries;
  }

  auto countEntries(SegmentFrontierKindSet kinds) const -> std::size_t
  {
    static constexpr std::array<SegmentFrontierKind, 3> frontier_kinds
        = {SegmentFrontierKind::kAll, SegmentFrontierKind::kTerminalBranchBuffered, SegmentFrontierKind::kTerminalLeafUnbuffered};
    std::size_t total_entries = 0U;
    for (const auto kind : frontier_kinds) {
      if (kinds.contains(kind)) {
        const auto* frontier = find(kind);
        total_entries += frontier == nullptr ? 0U : frontier->size();
      }
    }
    return total_entries;
  }

 private:
  auto entries(SegmentFrontierKind kind) const -> const std::vector<SegmentChar>&
  {
    switch (kind) {
      case SegmentFrontierKind::kAll:
        return _all_frontier_entries;
      case SegmentFrontierKind::kTerminalBranchBuffered:
        return _branch_buffered_entries;
      case SegmentFrontierKind::kTerminalLeafUnbuffered:
        return _leaf_unbuffered_entries;
    }
    return _all_frontier_entries;
  }

  SegmentFrontierKindSet _built_kinds;
  std::vector<SegmentChar> _all_frontier_entries;
  std::vector<SegmentChar> _branch_buffered_entries;
  std::vector<SegmentChar> _leaf_unbuffered_entries;
};

class SegmentFrontierCatalog
{
 public:
  SegmentFrontierCatalog() = default;
  explicit SegmentFrontierCatalog(std::unordered_map<unsigned, SegmentCandidateFrontierSet> entry_sets_by_length)
      : _entry_sets_by_length(std::move(entry_sets_by_length))
  {
  }

  auto empty() const -> bool { return _entry_sets_by_length.empty(); }
  auto lengthCount() const -> std::size_t { return _entry_sets_by_length.size(); }

  auto find(unsigned length_idx, SegmentFrontierKind kind) const -> const std::vector<SegmentChar>*
  {
    const auto it = _entry_sets_by_length.find(length_idx);
    if (it == _entry_sets_by_length.end()) {
      return nullptr;
    }
    return it->second.find(kind);
  }

  auto require(unsigned length_idx, SegmentFrontierKind kind) const -> const std::vector<SegmentChar>&
  {
    const auto* frontier = find(length_idx, kind);
    LOG_FATAL_IF(frontier == nullptr) << "HTree: required segment frontier length/kind was not synthesized.";
    return *frontier;
  }

  auto countEntries(SegmentFrontierKindSet kinds) const -> std::size_t
  {
    std::size_t total_entries = 0U;
    for (const auto& [length_idx, entry_set] : _entry_sets_by_length) {
      (void) length_idx;
      total_entries += entry_set.countEntries(kinds);
    }
    return total_entries;
  }

 private:
  std::unordered_map<unsigned, SegmentCandidateFrontierSet> _entry_sets_by_length;
};

struct RequiredLengthStateKey
{
  std::vector<unsigned> pending_lengths;

  auto operator==(const RequiredLengthStateKey& rhs) const -> bool = default;
};

struct RequiredLengthStateKeyHash
{
  auto operator()(const RequiredLengthStateKey& key) const noexcept -> std::size_t
  {
    std::size_t hash_value = 0U;
    for (const unsigned length_idx : key.pending_lengths) {
      hash_value ^= std::hash<unsigned>{}(length_idx) + 0x9e3779b9U + (hash_value << 6U) + (hash_value >> 2U);
    }
    return hash_value;
  }
};

struct SegmentClosureSolution
{
  bool feasible = false;
  unsigned total_cost = 0U;
  std::unordered_map<unsigned, SegmentCandidateFrontierSet> synthesized_entry_sets;
};

}  // namespace icts::htree
