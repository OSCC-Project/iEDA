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
 * @file CharFeasibilityChecker.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Characterization buffer lookup and electrical feasibility checks.
 */

#include "characterization/builder/CharFeasibilityChecker.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilderImpl.hh"

namespace icts::char_builder::detail {

auto CharFeasibilityChecker::findCharacterizationBufferCell(const std::string& cell_master) const
    -> const ::icts::CharacterizationBufferCell*
{
  auto it = std::ranges::find_if(_impl._sorted_buffers, [&cell_master](const ::icts::CharacterizationBufferCell& buffer_cell) -> bool {
    return buffer_cell.cell_master == cell_master;
  });
  return it == _impl._sorted_buffers.end() ? nullptr : &(*it);
}

auto CharFeasibilityChecker::calcClockRouteWireCapPf(double wirelength_um) const -> double
{
  return std::max(0.0, wirelength_um) * _impl._clock_route_segment_rc.capacitance_per_um_pf;
}

auto CharFeasibilityChecker::analyzePatternFeasibility(const TopologyDesc& topo, const std::vector<std::string>& buf_masters) const
    -> PatternFeasibility
{
  if (_impl._sorted_buffers.empty()) {
    return {};
  }

  const auto* source_buffer = &_impl._sorted_buffers.back();
  const auto* sink_buffer = &_impl._sorted_buffers.front();
  double max_external_load_pf = std::numeric_limits<double>::infinity();

  for (std::size_t segment_index = 0; segment_index < topo.wire_segments_um.size(); ++segment_index) {
    const ::icts::CharacterizationBufferCell* driver_buffer = nullptr;
    if (segment_index == 0U) {
      driver_buffer = source_buffer;
    } else {
      driver_buffer = findCharacterizationBufferCell(buf_masters.at(segment_index - 1U));
    }
    if (driver_buffer == nullptr || driver_buffer->max_cap_pf <= 0.0) {
      return {};
    }

    const double wire_cap_pf = calcClockRouteWireCapPf(topo.wire_segments_um.at(segment_index));
    const bool is_last_segment = (segment_index + 1U == topo.wire_segments_um.size());
    const ::icts::CharacterizationBufferCell* next_buffer
        = is_last_segment ? sink_buffer : findCharacterizationBufferCell(buf_masters.at(segment_index));
    const double next_input_cap_pf = next_buffer != nullptr ? next_buffer->input_cap_pf : 0.0;
    if (!is_last_segment && next_input_cap_pf <= 0.0) {
      return {};
    }

    const double static_stage_cap_pf = wire_cap_pf + next_input_cap_pf;
    if (!is_last_segment && static_stage_cap_pf > driver_buffer->max_cap_pf + kCapFeasibilityEpsilonPf) {
      return {};
    }
    if (is_last_segment) {
      max_external_load_pf = std::min(max_external_load_pf, driver_buffer->max_cap_pf - static_stage_cap_pf);
    }
  }

  if (!std::isfinite(max_external_load_pf)) {
    return PatternFeasibility{.is_pattern_feasible = true, .max_load_pf = std::numeric_limits<double>::infinity()};
  }
  if (max_external_load_pf + kCapFeasibilityEpsilonPf < 0.0) {
    return {};
  }
  return PatternFeasibility{.is_pattern_feasible = true, .max_load_pf = max_external_load_pf};
}

}  // namespace icts::char_builder::detail
