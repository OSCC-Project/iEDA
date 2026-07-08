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
 * @file RootDriverCompensation.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-04
 * @brief H-tree root driver compensation pass contracts.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ValueLattice.hh"
#include "synthesis/htree/HTree.hh"

namespace icts {

class HTreeTopologyChar;
class SchemaWriter;
class Wrapper;
struct PatternId;
class Tree;
}  // namespace icts

namespace icts::htree {

struct BufferPatternLibrary;
struct DepthSearchBuild;
struct DiagnosticBuild;
struct TopologyPatternLibrary;

struct RootDriverCompensationDetail
{
  bool enabled = false;
  bool valid = false;
  std::string method;
  std::string cell_master;
  std::string load_source;
  std::string route_estimator;
  double input_slew_ns = 0.0;
  unsigned load_bucket_idx = 0U;
  double load_cap_pf = 0.0;
  unsigned source_boundary_bucket_idx = 0U;
  double source_boundary_load_cap_pf = 0.0;
  std::size_t source_boundary_branch_count = 0U;
  double terminal_pin_cap_pf = 0.0;
  double wire_cap_pf = 0.0;
  double routed_wirelength_um = 0.0;
  std::size_t terminal_count = 0U;
  double clock_period_ns = 0.0;
  double output_slew_ns = 0.0;
  unsigned output_slew_bucket_idx = 0U;
  double cell_delay_ns = 0.0;
  double internal_power_w = 0.0;
  double leakage_power_w = 0.0;
  double cell_power_w = 0.0;
};

struct RootDriverCompensationInput
{
  bool enabled = false;
  Wrapper* wrapper = nullptr;
  double input_slew_ns = 0.0;
  double clock_period_ns = 0.0;
  UniformValueLattice cap_lattice;
  UniformValueLattice slew_lattice;
  std::string default_cell_master;
  int routing_layer = 0;
  std::optional<double> wire_width_um = std::nullopt;
  std::int32_t dbu_per_um = 0;
  SchemaWriter* reporter = nullptr;
  bool strict_boundary_closure = false;
  bool strict_slew_boundary_closure = true;
};

struct RootDriverCompensationStats
{
  bool enabled = false;
  std::string method;
  double input_slew_ns = 0.0;
  double clock_period_ns = 0.0;
  std::string load_source;
  std::size_t compensated_candidate_count = 0U;
  std::size_t boundary_input_candidate_count = 0U;
  std::size_t boundary_closed_candidate_count = 0U;
  std::size_t boundary_rejected_candidate_count = 0U;
  std::size_t boundary_cap_bucket_mismatch_count = 0U;
  std::size_t boundary_slew_bucket_mismatch_count = 0U;
  std::size_t invalid_compensation_count = 0U;
  std::size_t unique_direct_lookup_count = 0U;
  std::size_t cache_hit_count = 0U;
  std::size_t load_resolution_count = 0U;
  std::size_t load_resolution_cache_hit_count = 0U;
  std::size_t load_resolution_failure_count = 0U;
  std::size_t flute_route_estimate_count = 0U;
  std::size_t hpwl_route_estimate_count = 0U;
  double load_resolution_runtime_ms = 0.0;
  double total_runtime_ms = 0.0;
};

struct RootDriverBoundaryClosureCheck
{
  bool compensation_valid = false;
  bool cap_bucket_matches = false;
  bool slew_bucket_matches = false;
  unsigned raw_cap_bucket_idx = 0U;
  unsigned physical_load_bucket_idx = 0U;
  unsigned physical_source_boundary_bucket_idx = 0U;
  unsigned raw_input_slew_idx = 0U;
  unsigned root_output_slew_bucket_idx = 0U;
  RootDriverCompensationDetail compensation;

  auto isClosed(bool require_slew_bucket_match) const -> bool
  {
    return compensation_valid && cap_bucket_matches && (!require_slew_bucket_match || slew_bucket_matches);
  }
};

struct RootDriverCompensationApplySummary
{
  std::size_t input_candidate_count = 0U;
  std::size_t closed_candidate_count = 0U;
  std::size_t rejected_candidate_count = 0U;
  bool has_first_rejected_boundary = false;
  RootDriverBoundaryClosureCheck first_rejected_boundary;
};

class RootDriverCompensationPass
{
 public:
  explicit RootDriverCompensationPass(RootDriverCompensationInput input);
  ~RootDriverCompensationPass();

  RootDriverCompensationPass(const RootDriverCompensationPass&) = delete;
  auto operator=(const RootDriverCompensationPass&) -> RootDriverCompensationPass& = delete;
  RootDriverCompensationPass(RootDriverCompensationPass&&) noexcept;
  auto operator=(RootDriverCompensationPass&&) noexcept -> RootDriverCompensationPass&;

  auto beginCandidateBuild() -> void;
  auto apply(std::vector<HTreeTopologyChar>& entries, const TopologyPatternLibrary& topology_library,
             const BufferPatternLibrary& segment_pattern_library, const Tree& topology) -> RootDriverCompensationApplySummary;
  auto evaluate(PatternId pattern_id, const TopologyPatternLibrary& topology_library, const BufferPatternLibrary& segment_pattern_library,
                const Tree& topology) -> RootDriverCompensationDetail;
  auto get_stats() const -> const RootDriverCompensationStats&;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

inline constexpr double kRootDriverCompensationClockPeriodNs = 10.0;

auto ResolveRootDriverCompensationInputSlewNs(const HTree::Config& config, double max_slew_ns) -> double;
auto ResolveRootDriverClockPeriod(const HTree::Input& input) -> std::pair<double, std::string>;
auto ApplyRootDriverCompensationSummary(htree::DiagnosticBuild& build, const RootDriverCompensationStats& compensation_stats,
                                        const RootDriverCompensationDetail& compensation_detail, const HTreeTopologyChar& selected_entry)
    -> void;
auto ApplyRootDriverCompensationSummary(htree::DiagnosticBuild& build, const DepthSearchBuild& exploration,
                                        const RootDriverCompensationDetail& compensation_detail, const HTreeTopologyChar& selected_entry)
    -> void;

}  // namespace icts::htree
