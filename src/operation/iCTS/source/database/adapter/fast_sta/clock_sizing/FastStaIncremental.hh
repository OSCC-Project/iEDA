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
 * @file FastStaIncremental.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Incremental update coordinator for CTS fast STA contexts.
 */

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "FastSta.hh"
#include "clock_sizing/FastStaClockSizingEdit.hh"

namespace icts {

struct FastStaClockContext;

class FastStaIncremental
{
 public:
  FastStaIncremental() = delete;

  static auto changeBufferMaster(FastStaClockContext& context, FastStaNodeId node_id, std::string_view cell_master) -> bool;
  static auto changeBufferMasters(FastStaClockContext& context, const std::vector<FastStaBufferMasterChange>& changes) -> bool;
  static auto changeBufferMasterIncremental(FastStaClockContext& context, FastStaNodeId node_id, std::string_view cell_master)
      -> std::optional<FastStaDirtyRegion>;
};

}  // namespace icts
