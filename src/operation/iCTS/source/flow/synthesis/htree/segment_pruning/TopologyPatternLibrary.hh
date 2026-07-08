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
 * @file TopologyPatternLibrary.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief H-tree topology-pattern lookup, materialization, fanout legality, and composition.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "HTreeTopologyPattern.hh"
#include "PatternId.hh"
#include "characterization/Characterization.hh"
#include "log/Log.hh"

namespace icts::htree {

enum class TopologyPatternNodeKind
{
  kSeed,
  kConcat,
};

struct TopologyPatternNode
{
  PatternId pattern_id{PatternDomain::kTopologyPattern, 0U};
  unsigned levels = 0U;
  TerminalSemantic terminal_semantic = TerminalSemantic::kLeafUnbuffered;
  MonotonicBoundaryState monotonic_boundary_state{};
  std::size_t source_exposed_load_count = 1U;
  TopologyPatternNodeKind kind = TopologyPatternNodeKind::kSeed;
  PatternId segment_pattern_id{PatternDomain::kSegmentPattern, 0U};
  PatternId upstream_pattern_id{PatternDomain::kTopologyPattern, 0U};
  PatternId downstream_pattern_id{PatternDomain::kTopologyPattern, 0U};
};

struct TopologyPatternLibrary
{
  auto addSeed(PatternId pattern_id, PatternId segment_pattern_id, const PatternCompositionState& composition_state) -> void
  {
    LOG_FATAL_IF(pattern_id.domain != PatternDomain::kTopologyPattern) << "HTree: topology library received a non-topology pattern ID.";
    LOG_FATAL_IF(pattern_id.local_id != nodes.size()) << "HTree: topology library requires sequential topology pattern IDs.";
    nodes.push_back(TopologyPatternNode{
        .pattern_id = pattern_id,
        .levels = 1U,
        .terminal_semantic = composition_state.terminal_semantic,
        .monotonic_boundary_state = composition_state.monotonic_boundary_state,
        .source_exposed_load_count = composition_state.source_exposed_load_count,
        .kind = TopologyPatternNodeKind::kSeed,
        .segment_pattern_id = segment_pattern_id,
    });
  }

  auto addConcat(PatternId pattern_id, unsigned levels, PatternId upstream_pattern_id, PatternId downstream_pattern_id,
                 const PatternCompositionState& composition_state) -> void
  {
    LOG_FATAL_IF(pattern_id.domain != PatternDomain::kTopologyPattern) << "HTree: topology library received a non-topology pattern ID.";
    LOG_FATAL_IF(pattern_id.local_id != nodes.size()) << "HTree: topology library requires sequential topology pattern IDs.";
    nodes.push_back(TopologyPatternNode{
        .pattern_id = pattern_id,
        .levels = levels,
        .terminal_semantic = composition_state.terminal_semantic,
        .monotonic_boundary_state = composition_state.monotonic_boundary_state,
        .source_exposed_load_count = composition_state.source_exposed_load_count,
        .kind = TopologyPatternNodeKind::kConcat,
        .upstream_pattern_id = upstream_pattern_id,
        .downstream_pattern_id = downstream_pattern_id,
    });
  }

  auto findNode(PatternId pattern_id) const -> const TopologyPatternNode*
  {
    if (pattern_id.domain != PatternDomain::kTopologyPattern || pattern_id.local_id >= nodes.size()) {
      return nullptr;
    }
    return &nodes.at(pattern_id.local_id);
  }

  auto getCompositionState(PatternId pattern_id) const -> PatternCompositionState
  {
    const auto* node = findNode(pattern_id);
    LOG_FATAL_IF(node == nullptr) << "HTree: missing topology pattern composition-state cache entry.";
    return PatternCompositionState{
        .terminal_semantic = node->terminal_semantic,
        .monotonic_boundary_state = node->monotonic_boundary_state,
        .source_exposed_load_count = node->source_exposed_load_count,
    };
  }

  auto getTerminalSemantic(PatternId pattern_id) const -> TerminalSemantic { return getCompositionState(pattern_id).terminal_semantic; }

  auto materialize(PatternId pattern_id) const -> HTreeTopologyPattern
  {
    const auto* node = findNode(pattern_id);
    LOG_FATAL_IF(node == nullptr) << "HTree: missing topology pattern metadata.";

    std::vector<PatternId> level_segment_pattern_ids;
    level_segment_pattern_ids.reserve(node->levels);

    std::vector<PatternId> pending_pattern_ids;
    pending_pattern_ids.push_back(pattern_id);
    while (!pending_pattern_ids.empty()) {
      const PatternId current_pattern_id = pending_pattern_ids.back();
      pending_pattern_ids.pop_back();

      const auto* current_node = findNode(current_pattern_id);
      LOG_FATAL_IF(current_node == nullptr) << "HTree: missing topology pattern metadata during materialization.";

      if (current_node->kind == TopologyPatternNodeKind::kSeed) {
        level_segment_pattern_ids.push_back(current_node->segment_pattern_id);
        continue;
      }

      pending_pattern_ids.push_back(current_node->downstream_pattern_id);
      pending_pattern_ids.push_back(current_node->upstream_pattern_id);
    }

    LOG_FATAL_IF(level_segment_pattern_ids.size() != node->levels)
        << "HTree: materialized topology pattern level count does not match node metadata.";
    return HTreeTopologyPattern(pattern_id, node->levels, std::move(level_segment_pattern_ids));
  }

  auto compactReachableFrom(const std::vector<PatternId>& root_pattern_ids) const
      -> std::pair<TopologyPatternLibrary, std::unordered_map<PatternId, PatternId>>
  {
    std::vector<char> retained(nodes.size(), 0);
    std::vector<PatternId> pending_pattern_ids = root_pattern_ids;
    while (!pending_pattern_ids.empty()) {
      const PatternId pattern_id = pending_pattern_ids.back();
      pending_pattern_ids.pop_back();
      const auto* node = findNode(pattern_id);
      LOG_FATAL_IF(node == nullptr) << "HTree: missing topology pattern during reachable-library compaction.";
      if (retained.at(pattern_id.local_id) != 0) {
        continue;
      }
      retained.at(pattern_id.local_id) = 1;
      if (node->kind == TopologyPatternNodeKind::kConcat) {
        pending_pattern_ids.push_back(node->downstream_pattern_id);
        pending_pattern_ids.push_back(node->upstream_pattern_id);
      }
    }

    TopologyPatternLibrary compact_library;
    std::unordered_map<PatternId, PatternId> pattern_id_map;
    pattern_id_map.reserve(root_pattern_ids.size());
    for (std::size_t node_index = 0U; node_index < nodes.size(); ++node_index) {
      if (retained.at(node_index) == 0) {
        continue;
      }
      const PatternId old_pattern_id = PatternId::topology(static_cast<unsigned>(node_index));
      const PatternId new_pattern_id = PatternId::topology(static_cast<unsigned>(compact_library.nodes.size()));
      pattern_id_map.emplace(old_pattern_id, new_pattern_id);
      compact_library.nodes.push_back(nodes.at(node_index));
      compact_library.nodes.back().pattern_id = new_pattern_id;
    }

    for (auto& node : compact_library.nodes) {
      if (node.kind != TopologyPatternNodeKind::kConcat) {
        continue;
      }
      const auto upstream_it = pattern_id_map.find(node.upstream_pattern_id);
      const auto downstream_it = pattern_id_map.find(node.downstream_pattern_id);
      LOG_FATAL_IF(upstream_it == pattern_id_map.end() || downstream_it == pattern_id_map.end())
          << "HTree: compacted topology pattern lost a reachable child pattern.";
      node.upstream_pattern_id = upstream_it->second;
      node.downstream_pattern_id = downstream_it->second;
    }

    return {std::move(compact_library), std::move(pattern_id_map)};
  }

  std::vector<TopologyPatternNode> nodes;
};

struct PatternSearchBuild
{
  bool success = false;
  std::string failure_reason;
  unsigned failure_level = 0U;
  unsigned failure_length_idx = 0U;
  std::vector<HTreeTopologyChar> frontier;
  TopologyPatternLibrary topology_pattern_library;
};

inline auto IsBinarySourceFanoutLegal(std::size_t per_branch_load_count, std::size_t max_fanout) -> bool
{
  if (max_fanout == 0U) {
    return true;
  }
  if (per_branch_load_count > std::numeric_limits<std::size_t>::max() / 2U) {
    return false;
  }
  return per_branch_load_count * 2U <= max_fanout;
}

class TopologyPatternLibraryCombiner
{
 public:
  TopologyPatternLibraryCombiner(TopologyPatternLibrary& library, unsigned start_id, std::size_t max_fanout)
      : _library(&library), _next_id(start_id), _max_fanout(max_fanout)
  {
  }

  auto canCompose(PatternId upstream, PatternId downstream) const -> bool
  {
    const auto upstream_state = _library->getCompositionState(upstream);
    const auto downstream_state = _library->getCompositionState(downstream);
    return upstream_state.monotonic_boundary_state.canComposeWith(downstream_state.monotonic_boundary_state)
           && isBranchFanoutLegal(downstream_state);
  }

  auto composeState(PatternId upstream, PatternId downstream) const -> PatternCompositionState
  {
    const auto upstream_state = _library->getCompositionState(upstream);
    const auto downstream_state = _library->getCompositionState(downstream);
    return PatternCompositionState{
        .terminal_semantic = downstream_state.terminal_semantic,
        .monotonic_boundary_state
        = MonotonicBoundaryState::compose(upstream_state.monotonic_boundary_state, downstream_state.monotonic_boundary_state),
        .source_exposed_load_count = resolveMergedSourceLoadCount(upstream_state, downstream_state),
    };
  }

  auto combine(PatternId upstream, PatternId downstream) const -> PatternId
  {
    const auto* upstream_pattern = _library->findNode(upstream);
    const auto* downstream_pattern = _library->findNode(downstream);
    LOG_FATAL_IF(upstream_pattern == nullptr || downstream_pattern == nullptr) << "HTree: missing topology pattern during composition.";
    LOG_FATAL_IF(!canCompose(upstream, downstream)) << "HTree: invalid non-monotonic topology pattern composition.";

    const PatternId merged_pattern_id = PatternId::topology(_next_id++);
    _library->addConcat(merged_pattern_id, upstream_pattern->levels + downstream_pattern->levels, upstream, downstream,
                        composeState(upstream, downstream));
    return merged_pattern_id;
  }

  auto get_next_id() const -> unsigned { return _next_id; }

 private:
  auto isBranchFanoutLegal(const PatternCompositionState& downstream_state) const -> bool
  {
    return IsBinarySourceFanoutLegal(downstream_state.source_exposed_load_count, _max_fanout);
  }

  static auto resolveMergedSourceLoadCount(const PatternCompositionState& upstream_state, const PatternCompositionState& downstream_state)
      -> std::size_t
  {
    if (upstream_state.monotonic_boundary_state.source.has_buffer) {
      return 1U;
    }
    if (downstream_state.source_exposed_load_count > std::numeric_limits<std::size_t>::max() / 2U) {
      return std::numeric_limits<std::size_t>::max();
    }
    return downstream_state.source_exposed_load_count * 2U;
  }

  TopologyPatternLibrary* _library = nullptr;
  mutable unsigned _next_id;
  std::size_t _max_fanout = 0U;
};

}  // namespace icts::htree
