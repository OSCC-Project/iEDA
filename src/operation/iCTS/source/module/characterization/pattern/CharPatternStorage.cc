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
 * @file CharPatternStorage.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Characterization buffering-pattern metadata storage.
 */

#include "characterization/pattern/CharPatternStorage.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "PatternId.hh"
#include "ValueLattice.hh"
#include "characterization/builder/CharBuilderImpl.hh"

namespace icts::char_builder::detail {

auto CharPatternStorage::storeBufferingPattern(unsigned length_idx, const TopologyDesc& topo, const std::vector<std::string>& buf_masters,
                                               double total_length_um) -> ::icts::PatternId
{
  std::vector<double> buffer_positions_norm;
  if (!topo.buffer_positions.empty() && total_length_um > 0.0) {
    double cumulative_um = 0.0;
    size_t buf_idx = 0;
    for (size_t seg = 0; seg < topo.wire_segments_um.size() && buf_idx < topo.buffer_positions.size(); ++seg) {
      cumulative_um += topo.wire_segments_um.at(seg);
      if (seg < topo.wire_segments_um.size() - 1) {
        const double normalized = cumulative_um / total_length_um;
        buffer_positions_norm.push_back(normalized);
        ++buf_idx;
      }
    }
  }
  if (topo.has_terminal_branch_buffer
      && (buffer_positions_norm.empty() || std::abs(buffer_positions_norm.back() - 1.0) > ::icts::kValueLatticeEpsilon)) {
    buffer_positions_norm.push_back(1.0);
  }

  const ::icts::PatternId pid = ::icts::PatternId::segment(_impl._next_pattern_id);
  ::icts::BufferingPattern pattern(length_idx, pid, buffer_positions_norm, buf_masters, topo.has_terminal_branch_buffer);
  _impl._buffering_patterns.push_back(std::move(pattern));
  return pid;
}

}  // namespace icts::char_builder::detail
