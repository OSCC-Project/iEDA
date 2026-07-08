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
 * @file CharStaSampler.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder per-pattern STA-sampling pipeline component.
 *        Receives one candidate (topology, buffer-masters) pair and walks the
 *        load * input-slew sweep through fast-STA, storing each successful
 *        sample as a SegmentChar entry with lattice-validated indices.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace icts {
struct PatternId;
}  // namespace icts

namespace icts::char_builder::detail {

class CharBuilderImpl;
struct TopologyDesc;
struct BuildProgress;
struct PatternFeasibility;
struct StoredSampleIndices;

class CharStaSampler
{
 public:
  explicit CharStaSampler(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharStaSampler() = default;
  CharStaSampler(const CharStaSampler&) = delete;
  auto operator=(const CharStaSampler&) -> CharStaSampler& = delete;

  auto characterizeTopology(unsigned length_idx, const TopologyDesc& topo, const std::vector<std::string>& buf_masters,
                            BuildProgress& build_progress) -> void;

 private:
  auto sampleFeasibleTopology(unsigned length_idx, const ::icts::PatternId& pid, const TopologyDesc& topo,
                              const std::vector<std::string>& buf_masters, const PatternFeasibility& feasibility,
                              BuildProgress& build_progress) -> void;
  auto sampleLoadSlews(unsigned length_idx, const ::icts::PatternId& pid, const TopologyDesc& topo, double effective_load_pf,
                       double load_pf, double driven_cap_pf, bool& power_context_ready, BuildProgress& build_progress) -> void;
  auto tryMakeStoredSampleIndices(unsigned input_slew_idx, unsigned load_cap_idx, double output_slew_ns, double driven_cap_pf,
                                  BuildProgress& build_progress) const -> std::optional<StoredSampleIndices>;

  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
