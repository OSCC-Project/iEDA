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
 * @file CharPatternEnumerator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Buffer-topology and monotonic buffer-combination enumeration.
 */

#include "characterization/pattern/CharPatternEnumerator.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <vector>

#include "Log.hh"
#include "ValueLattice.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilderImpl.hh"
#include "characterization/builder/CharTopologyPlanner.hh"
#include "characterization/sampling/CharStaSampler.hh"

namespace icts::char_builder::detail {
namespace {

constexpr unsigned kMaxTopologySlots = std::numeric_limits<std::uint64_t>::digits - 1U;

}  // namespace

auto CharPatternEnumerator::calcTopologySlotCount(double wirelength_um) const -> unsigned
{
  const ::icts::UniformValueLattice length_lattice(_impl._length_unit_um, _impl._wirelength_iterations);
  const auto length_idx = length_lattice.tryObservedIndex(wirelength_um);
  auto slot_count = length_idx.value_or(length_lattice.coveringIndex(wirelength_um));
  if (slot_count > kMaxTopologySlots) {
    static bool has_logged_slot_clamp = false;
    if (!has_logged_slot_clamp) {
      LOG_WARNING << "CharBuilder: slot count exceeds topology bit capacity, clamp to " << kMaxTopologySlots;
      has_logged_slot_clamp = true;
    }
    slot_count = kMaxTopologySlots;
  }
  return slot_count;
}

auto CharPatternEnumerator::countSelectedSlots(TopologyBits topology_bits) -> unsigned
{
  unsigned slot_count = 0U;
  auto remaining_bits = topology_bits.value;
  while (remaining_bits != 0U) {
    slot_count += static_cast<unsigned>(remaining_bits & 1U);
    remaining_bits >>= 1U;
  }
  return slot_count;
}

auto CharPatternEnumerator::estimatePatternCountPerWirelength(double wirelength_um) const -> std::size_t
{
  const unsigned num_slots = calcTopologySlotCount(wirelength_um);
  LOG_FATAL_IF(num_slots >= std::numeric_limits<std::uint64_t>::digits)
      << "CharBuilder: buffer slot count " << num_slots << " exceeds topology bit capacity.";

  std::size_t total_patterns = 0;
  const std::uint64_t num_topologies = std::uint64_t{1} << num_slots;
  for (std::uint64_t topology_bits_value = 0; topology_bits_value < num_topologies; ++topology_bits_value) {
    const unsigned num_buffer_positions = countSelectedSlots(TopologyBits{topology_bits_value});
    total_patterns += (num_buffer_positions == 0U) ? 1U : getMonotonicComboCount(_impl._sorted_buffers.size(), num_buffer_positions);
  }
  return total_patterns;
}

auto CharPatternEnumerator::enumerateWirelength(unsigned length_idx, double wirelength_um, BuildProgress& build_progress) -> void
{
  const unsigned num_slots = calcTopologySlotCount(wirelength_um);
  LOG_FATAL_IF(num_slots >= std::numeric_limits<std::uint64_t>::digits)
      << "CharBuilder: buffer slot count " << num_slots << " exceeds topology bit capacity.";

  const std::uint64_t num_topologies = std::uint64_t{1} << num_slots;
  for (std::uint64_t topology_bits_value = 0; topology_bits_value < num_topologies; ++topology_bits_value) {
    enumerateTopology(length_idx, wirelength_um, num_slots, TopologyBits{topology_bits_value}, build_progress);
  }
}

auto CharPatternEnumerator::enumerateTopology(unsigned length_idx, double wirelength_um, unsigned num_slots, TopologyBits topology_bits,
                                              BuildProgress& build_progress) -> void
{
  const TopologyDesc topo = _impl.topologyPlanner().buildTopologyDesc(wirelength_um, num_slots, topology_bits);
  const std::size_t num_buf_positions = topo.buffer_positions.size();

  if (num_buf_positions == 0) {
    const std::vector<std::string> empty_masters;
    _impl.staSampler().characterizeTopology(length_idx, topo, empty_masters, build_progress);
    return;
  }

  const std::size_t num_buf_types = _impl._sorted_buffers.size();
  if (num_buf_types == 0) {
    return;
  }

  std::vector<std::size_t> buf_indices(num_buf_positions, 0);
  while (true) {
    std::vector<std::string> buf_masters;
    buf_masters.reserve(num_buf_positions);
    for (const auto buffer_index : std::ranges::reverse_view(buf_indices)) {
      buf_masters.push_back(_impl._sorted_buffers.at(buffer_index).cell_master);
    }

    _impl.staSampler().characterizeTopology(length_idx, topo, buf_masters, build_progress);
    if (!advanceToNextMonotonic(buf_indices, num_buf_types)) {
      break;
    }
  }
}

auto CharPatternEnumerator::getMonotonicComboCount(std::size_t num_buf_types, std::size_t num_positions) -> std::size_t
{
  if (num_buf_types == 0 || num_positions == 0) {
    return 0;
  }
  const std::size_t combination_n = num_buf_types + num_positions - 1;
  std::size_t combination_k = num_positions;
  combination_k = std::min(combination_k, combination_n - combination_k);
  std::size_t result = 1;
  for (std::size_t index = 0; index < combination_k; ++index) {
    result = result * (combination_n - index) / (index + 1);
  }
  return result;
}

auto CharPatternEnumerator::advanceToNextMonotonic(std::vector<std::size_t>& buf_indices, std::size_t num_buf_types) -> bool
{
  if (buf_indices.empty() || num_buf_types == 0) {
    return false;
  }

  int position = static_cast<int>(buf_indices.size()) - 1;

  while (position >= 0) {
    const auto index = static_cast<std::size_t>(position);
    if (buf_indices.at(index) + 1 < num_buf_types) {
      ++buf_indices.at(index);
      for (std::size_t tail_index = index + 1; tail_index < buf_indices.size(); ++tail_index) {
        buf_indices.at(tail_index) = buf_indices.at(index);
      }
      return true;
    }
    --position;
  }

  return false;
}

}  // namespace icts::char_builder::detail
