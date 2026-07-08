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
 * @file BufferStrengthTable.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree pattern-side buffer drive-strength ranking table.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/Wrapper.hh"
#include "log/Log.hh"

namespace icts::htree {

class BufferStrengthTable
{
 public:
  explicit BufferStrengthTable(Wrapper& wrapper) : _wrapper(&wrapper) {}

  auto getStrengthRank(const std::string& cell_master) -> unsigned
  {
    if (cell_master.empty()) {
      return 0U;
    }

    if (!_drive_caps.contains(cell_master)) {
      LOG_FATAL_IF(_wrapper == nullptr) << "HTree: Wrapper is unavailable for buffer drive-strength ranking.";
      double drive_cap_pf = _wrapper->queryCellOutPinCapLimit(cell_master);
      if (drive_cap_pf <= 0.0) {
        drive_cap_pf = _wrapper->queryCellOutPinCapTableAxisMax(cell_master);
      }
      _drive_caps[cell_master] = drive_cap_pf;
      _ranks_dirty = true;

      if (drive_cap_pf <= 0.0) {
        LOG_WARNING << "HTree: failed to resolve drive-strength rank for buffer master " << cell_master
                    << "; monotonic composition keeps an explicit boundary buffer with unresolved size class.";
      }
    }

    if (_ranks_dirty) {
      rebuildRanks();
    }

    const auto rank_it = _strength_ranks.find(cell_master);
    return rank_it == _strength_ranks.end() ? 0U : rank_it->second;
  }

 private:
  auto rebuildRanks() -> void
  {
    std::vector<std::pair<std::string, double>> ordered_caps;
    ordered_caps.reserve(_drive_caps.size());
    for (const auto& [cell_master, drive_cap_pf] : _drive_caps) {
      if (drive_cap_pf > 0.0) {
        ordered_caps.emplace_back(cell_master, drive_cap_pf);
      }
    }

    std::ranges::sort(ordered_caps, [](const auto& lhs, const auto& rhs) -> bool {
      if (lhs.second != rhs.second) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    });

    _strength_ranks.clear();
    unsigned current_rank = 0U;
    double last_cap_pf = 0.0;
    bool has_last_cap = false;
    for (const auto& [cell_master, drive_cap_pf] : ordered_caps) {
      if (!has_last_cap || std::abs(drive_cap_pf - last_cap_pf) > 1e-12) {
        ++current_rank;
        last_cap_pf = drive_cap_pf;
        has_last_cap = true;
      }
      _strength_ranks[cell_master] = current_rank;
    }

    _ranks_dirty = false;
  }

  std::unordered_map<std::string, double> _drive_caps;
  std::unordered_map<std::string, unsigned> _strength_ranks;
  Wrapper* _wrapper = nullptr;
  bool _ranks_dirty = false;
};

}  // namespace icts::htree
