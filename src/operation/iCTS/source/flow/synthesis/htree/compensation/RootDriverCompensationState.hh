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
 * @file RootDriverCompensationState.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief H-tree root-driver compensation state and root-load estimate contracts.
 */

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "PatternId.hh"
#include "Point.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"

namespace icts {
class Tree;
}  // namespace icts

namespace icts::htree {

inline constexpr const char* kRootDriverCompensationMethod = "direct";
inline constexpr const char* kRootDriverCompensationLoadSource = "root_closure_physical_estimate";

struct RootClosureTerminal
{
  std::string name;
  Point<int> location;
  double pin_cap_pf = 0.0;
};

struct RootClosureWireEstimate
{
  std::string route_estimator;
  double wire_cap_pf = 0.0;
  double routed_wirelength_um = 0.0;
};

struct RootDriverCompensationCacheKey
{
  std::string cell_master;
  double input_slew_ns = 0.0;
  unsigned load_bucket_idx = 0U;
  double load_cap_pf = 0.0;
  double clock_period_ns = 0.0;

  auto operator==(const RootDriverCompensationCacheKey& rhs) const -> bool = default;
};

struct RootDriverCompensationCacheKeyHash
{
  auto operator()(const RootDriverCompensationCacheKey& key) const noexcept -> std::size_t;
};

struct RootClosureLoadEstimate
{
  bool valid = false;
  std::string source;
  std::string route_estimator;
  unsigned bucket_idx = 0U;
  double total_load_cap_pf = 0.0;
  unsigned source_boundary_bucket_idx = 0U;
  double source_boundary_load_cap_pf = 0.0;
  std::size_t source_boundary_branch_count = 0U;
  double terminal_pin_cap_pf = 0.0;
  double wire_cap_pf = 0.0;
  double routed_wirelength_um = 0.0;
  std::size_t terminal_count = 0U;
};

struct RootClosureLoadSignature
{
  bool ends_at_real_buffer = false;
  std::vector<PatternId> root_prefix_segment_pattern_ids;

  auto operator==(const RootClosureLoadSignature& rhs) const -> bool = default;
};

struct RootClosureLoadSignatureHash
{
  auto operator()(const RootClosureLoadSignature& signature) const noexcept -> std::size_t;
};

struct RootDriverCompensationState
{
  RootDriverCompensationInput input;
  std::unordered_map<RootDriverCompensationCacheKey, RootDriverCompensationDetail, RootDriverCompensationCacheKeyHash> cost_by_key;
  std::unordered_map<RootClosureLoadSignature, RootClosureLoadEstimate, RootClosureLoadSignatureHash> root_load_by_signature;
  RootDriverCompensationStats stats;
  bool warned_invalid_input = false;
};

auto QueryRootClosureLoadEstimate(PatternId pattern_id, const TopologyPatternLibrary& topology_library,
                                  const BufferPatternLibrary& segment_pattern_library, const Tree& topology,
                                  RootDriverCompensationState& state) -> RootClosureLoadEstimate;

}  // namespace icts::htree
