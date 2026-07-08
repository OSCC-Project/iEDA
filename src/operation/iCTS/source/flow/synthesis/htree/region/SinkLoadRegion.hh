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
 * @file SinkLoadRegion.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief H-tree sink-load-region legality data types.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "HTreeTopologyChar.hh"
#include "PatternId.hh"
#include "Point.hh"
#include "ValueLattice.hh"

namespace icts {

class Pin;
class Wrapper;
class Tree;

namespace htree {

struct BufferPatternLibrary;
struct TopologyPatternLibrary;

struct CapDistributionStats
{
  std::size_t group_count = 0U;
  double cap_min_pf = 0.0;
  double cap_max_pf = 0.0;
  double cap_mean_pf = 0.0;
  double cap_median_pf = 0.0;
};

struct SinkLoadRegionBoundaryGroup
{
  std::size_t node_id = std::numeric_limits<std::size_t>::max();
  Point<int> anchor;
  const std::vector<Pin*>* loads = nullptr;
};

struct SinkLoadRegionLegalitySignature
{
  int bottom_most_buffered_level = -1;
  PatternId segment_pattern_id = PatternId::segment(0);

  auto operator==(const SinkLoadRegionLegalitySignature& rhs) const -> bool = default;
};

struct SinkLoadRegionLegalitySignatureHash
{
  auto operator()(const SinkLoadRegionLegalitySignature& signature) const noexcept -> std::size_t
  {
    std::size_t hash_value = std::hash<int>{}(signature.bottom_most_buffered_level);
    hash_value ^= std::hash<unsigned>{}(signature.segment_pattern_id.pack()) + 0x9e3779b9U + (hash_value << 6U) + (hash_value >> 2U);
    return hash_value;
  }
};

enum class SinkLoadRegionViolation
{
  kNone,
  kMissingTopologyRoot,
  kMissingTopologyNode,
  kMissingTopologyLevel,
  kMissingSegmentPattern,
  kMissingBufferPosition,
  kEmptyLoadGroup,
  kFanout,
  kPinCapLowerBound,
  kRoutingFailed,
  kCapacitance,
};

struct SinkLoadRegionLegalitySummary
{
  bool legal = false;
  bool monotone_hard_fail = false;
  SinkLoadRegionViolation violation = SinkLoadRegionViolation::kMissingTopologyRoot;
  std::string failure_reason;
  CapDistributionStats cap_distribution;
  double required_leaf_load_cap_pf = 0.0;
  std::optional<unsigned> required_leaf_load_cap_covering_idx = std::nullopt;
  int bottom_most_buffered_level = -1;
  PatternId segment_pattern_id = PatternId::segment(0);
};

struct SinkLoadRegionLegalityInput
{
  Wrapper* wrapper = nullptr;
  std::size_t max_fanout = 0U;
  bool has_max_cap = false;
  double max_cap_pf = std::numeric_limits<double>::infinity();
  ClockRouteSegmentRc clock_route_segment_rc;
};

struct SinkLoadRegionLegalityContext
{
  std::unordered_map<SinkLoadRegionLegalitySignature, SinkLoadRegionLegalitySummary, SinkLoadRegionLegalitySignatureHash>
      result_by_signature;
  int max_monotone_failed_level = std::numeric_limits<int>::min();
  UniformValueLattice cap_lattice;
  SinkLoadRegionLegalityInput input;
};

struct SinkLoadRegionEntryFilterOutput
{
  std::vector<HTreeTopologyChar> entries;
};

struct SinkLoadRegionEntryFilterSummary
{
  std::string first_failure_reason;
};

struct SinkLoadRegionEntryFilterBuild
{
  SinkLoadRegionEntryFilterOutput output;
  SinkLoadRegionEntryFilterSummary summary;
};

auto ResolveSinkLoadRegionLegality(const Tree& topology, PatternId topology_pattern_id, const TopologyPatternLibrary& topology_library,
                                   const BufferPatternLibrary& segment_pattern_library, SinkLoadRegionLegalityContext& legality_context)
    -> SinkLoadRegionLegalitySummary;
auto FilterSinkLoadRegionLegalEntries(const std::vector<HTreeTopologyChar>& entries, const Tree& topology,
                                      const TopologyPatternLibrary& topology_library, const BufferPatternLibrary& segment_pattern_library,
                                      SinkLoadRegionLegalityContext& legality_context) -> SinkLoadRegionEntryFilterBuild;

}  // namespace htree
}  // namespace icts
