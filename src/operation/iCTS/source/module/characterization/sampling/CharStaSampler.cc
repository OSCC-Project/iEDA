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
 * @file CharStaSampler.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Per-topology STA sampling pipeline for the characterization sweep:
 *        guards by feasibility, walks load and input-slew sweeps, queries
 *        fast-STA, validates lattice indices, and emits SegmentChar entries.
 */

#include "characterization/sampling/CharStaSampler.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "CharCore.hh"
#include "FastSta.hh"
#include "Log.hh"
#include "PatternId.hh"
#include "SegmentChar.hh"
#include "ValueLattice.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "characterization/builder/CharFeasibilityChecker.hh"
#include "characterization/circuit/CharCircuitBuilder.hh"
#include "characterization/pattern/CharPatternStorage.hh"

namespace icts::char_builder::detail {
namespace {

constexpr std::size_t kCharProgressLogStride = 32U;
constexpr bool kEnableCharPowerSampling = true;

}  // namespace

auto CharStaSampler::characterizeTopology(unsigned length_idx, const TopologyDesc& topo, const std::vector<std::string>& buf_masters,
                                          BuildProgress& build_progress) -> void
{
  ++build_progress.evaluated_patterns;

  double total_length_um = 0.0;
  for (const double seg_len : topo.wire_segments_um) {
    total_length_um += seg_len;
  }

  const auto pid = _impl.patternStorage().storeBufferingPattern(length_idx, topo, buf_masters, total_length_um);

  const PatternFeasibility feasibility = _impl.feasibilityChecker().analyzePatternFeasibility(topo, buf_masters);
  if (!feasibility.is_pattern_feasible) {
    ++build_progress.skipped_patterns_infeasible;
    if ((build_progress.evaluated_patterns % kCharProgressLogStride) == 0U) {
      LOG_INFO << "CharBuilder: wirelength=" << total_length_um << " um progress " << build_progress.evaluated_patterns << "/"
               << build_progress.estimated_patterns << " patterns"
               << " (feasible=" << build_progress.feasible_patterns << ", skipped=" << build_progress.skipped_patterns_infeasible
               << ", executed_sta_samples=" << build_progress.executed_sta_samples << ")";
    }
    ++_impl._next_pattern_id;
    return;
  }
  ++build_progress.feasible_patterns;

  sampleFeasibleTopology(length_idx, pid, topo, buf_masters, feasibility, build_progress);
  if ((build_progress.evaluated_patterns % kCharProgressLogStride) == 0U) {
    LOG_INFO << "CharBuilder: wirelength=" << total_length_um << " um progress " << build_progress.evaluated_patterns << "/"
             << build_progress.estimated_patterns << " patterns"
             << " (feasible=" << build_progress.feasible_patterns << ", skipped=" << build_progress.skipped_patterns_infeasible
             << ", executed_sta_samples=" << build_progress.executed_sta_samples
             << ", skipped_sta_samples=" << build_progress.skipped_sta_samples << ")";
  }
  ++_impl._next_pattern_id;
}

auto CharStaSampler::sampleFeasibleTopology(unsigned length_idx, const ::icts::PatternId& pid, const TopologyDesc& topo,
                                            const std::vector<std::string>& buf_masters, const PatternFeasibility& feasibility,
                                            BuildProgress& build_progress) -> void
{
  _impl.circuitBuilder().createCharCircuit(topo, buf_masters);

  bool power_context_ready = kEnableCharPowerSampling;

  const ::icts::UniformValueLattice cap_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_cap, _impl._cap_steps);
  for (const double load_pf : _impl._loads_to_test) {
    if (load_pf > feasibility.max_load_pf + kCapFeasibilityEpsilonPf) {
      ++build_progress.skipped_load_points;
      build_progress.skipped_sta_samples += _impl._slews_to_test.size();
      continue;
    }

    const double effective_load = load_pf - _impl._sink_input_cap_pf;
    if (effective_load < 0.0) {
      ++build_progress.skipped_load_points;
      build_progress.skipped_sta_samples += _impl._slews_to_test.size();
      continue;
    }

    double driven_cap_pf = 0.0;
    if (!buf_masters.empty()) {
      const auto* first_buffer = _impl.feasibilityChecker().findCharacterizationBufferCell(buf_masters.front());
      if (first_buffer == nullptr || first_buffer->input_cap_pf <= 0.0) {
        ++build_progress.skipped_load_points;
        build_progress.skipped_sta_samples += _impl._slews_to_test.size();
        continue;
      }
      driven_cap_pf = first_buffer->input_cap_pf;
      driven_cap_pf += _impl.feasibilityChecker().calcClockRouteWireCapPf(topo.wire_segments_um.front());
    } else {
      driven_cap_pf = load_pf;
      for (const double seg_len_um : topo.wire_segments_um) {
        driven_cap_pf += _impl.feasibilityChecker().calcClockRouteWireCapPf(seg_len_um);
      }
    }

    build_progress.max_observed_driven_cap_pf = std::max(build_progress.max_observed_driven_cap_pf, driven_cap_pf);
    build_progress.max_observed_driven_cap_idx
        = std::max(build_progress.max_observed_driven_cap_idx, cap_lattice.coveringIndex(driven_cap_pf));
    const auto driven_cap_idx = cap_lattice.tryObservedIndex(driven_cap_pf);
    if (!driven_cap_idx.has_value()) {
      ++build_progress.skipped_load_points;
      build_progress.skipped_sta_samples += _impl._slews_to_test.size();
      build_progress.driven_cap_overflow_samples += _impl._slews_to_test.size();
      ++build_progress.driven_cap_overflow_load_points;
      continue;
    }

    sampleLoadSlews(length_idx, pid, topo, effective_load, load_pf, driven_cap_pf, power_context_ready, build_progress);
  }

  _impl.circuitBuilder().destroyCharCircuit();
}

auto CharStaSampler::sampleLoadSlews(unsigned length_idx, const ::icts::PatternId& pid, const TopologyDesc& topo, double effective_load_pf,
                                     double load_pf, double driven_cap_pf, bool& power_context_ready, BuildProgress& build_progress) -> void
{
  _impl.circuitBuilder().setCharParasitics(topo, effective_load_pf);

  const ::icts::UniformValueLattice slew_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_slew, _impl._slew_steps);
  const ::icts::UniformValueLattice cap_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_cap, _impl._cap_steps);
  const unsigned load_cap_idx = cap_lattice.coveringIndex(load_pf);
  for (const double input_slew_ns : _impl._slews_to_test) {
    LOG_FATAL_IF(_impl._fast_sta == nullptr) << "CharStaSampler: FastSTA dependency is not configured.";
    const auto sample_result = _impl._fast_sta->runCharSample(_impl._fast_sta_char_context_id, input_slew_ns);
    ++build_progress.executed_sta_samples;
    const unsigned input_slew_idx = slew_lattice.coveringIndex(input_slew_ns);

    double power_w = 0.0;
    double source_boundary_net_switch_power_w = 0.0;
    if (!sample_result.valid) {
      ++build_progress.skipped_sta_samples;
      continue;
    }
    if (kEnableCharPowerSampling && power_context_ready) {
      power_w = sample_result.power_w;
      source_boundary_net_switch_power_w = sample_result.source_boundary_net_switch_power_w;
    }

    const auto stored_sample_indices
        = tryMakeStoredSampleIndices(input_slew_idx, load_cap_idx, sample_result.output_slew_ns, driven_cap_pf, build_progress);
    if (!stored_sample_indices.has_value()) {
      continue;
    }

    const ::icts::CharCore core(stored_sample_indices->input_slew_idx, stored_sample_indices->output_slew_idx,
                                stored_sample_indices->driven_cap_idx, stored_sample_indices->load_cap_idx, sample_result.delay_ns, power_w,
                                pid, source_boundary_net_switch_power_w);
    const ::icts::SegmentChar seg_char(core, length_idx);
    _impl._segment_chars.push_back(seg_char);
  }
}

auto CharStaSampler::tryMakeStoredSampleIndices(unsigned input_slew_idx, unsigned load_cap_idx, double output_slew_ns, double driven_cap_pf,
                                                BuildProgress& build_progress) const -> std::optional<StoredSampleIndices>
{
  const ::icts::UniformValueLattice slew_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_slew, _impl._slew_steps);
  const ::icts::UniformValueLattice cap_lattice = ::icts::UniformValueLattice::buildFromMax(_impl._max_cap, _impl._cap_steps);
  const unsigned output_slew_idx = slew_lattice.coveringIndex(output_slew_ns);
  const unsigned driven_cap_idx = cap_lattice.coveringIndex(driven_cap_pf);

  build_progress.max_observed_output_slew_ns = std::max(build_progress.max_observed_output_slew_ns, output_slew_ns);
  build_progress.max_observed_output_slew_idx = std::max(build_progress.max_observed_output_slew_idx, output_slew_idx);
  build_progress.max_observed_driven_cap_pf = std::max(build_progress.max_observed_driven_cap_pf, driven_cap_pf);
  build_progress.max_observed_driven_cap_idx = std::max(build_progress.max_observed_driven_cap_idx, driven_cap_idx);

  const auto observed_output_slew_idx = slew_lattice.tryObservedIndex(output_slew_ns);
  if (!observed_output_slew_idx.has_value()) {
    ++build_progress.output_slew_overflow_samples;
    return std::nullopt;
  }
  const auto observed_driven_cap_idx = cap_lattice.tryObservedIndex(driven_cap_pf);
  if (!observed_driven_cap_idx.has_value()) {
    ++build_progress.driven_cap_overflow_samples;
    return std::nullopt;
  }

  return StoredSampleIndices{
      .input_slew_idx = input_slew_idx,
      .output_slew_idx = *observed_output_slew_idx,
      .driven_cap_idx = *observed_driven_cap_idx,
      .load_cap_idx = load_cap_idx,
  };
}

}  // namespace icts::char_builder::detail
