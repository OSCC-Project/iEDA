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
 * @file FastStaTiming.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS fast STA timing propagation.
 */

#pragma once

#include "FastSta.hh"

namespace icts {

struct FastStaClockContext;
struct FastStaDirtyRegion;

class FastStaTiming
{
 public:
  FastStaTiming() = delete;

  static auto update(FastStaClockContext& context) -> bool;
  static auto updateRegion(FastStaClockContext& context, const FastStaDirtyRegion& dirty_region) -> bool;
  static auto calcSkew(const FastStaClockContext& context) -> FastStaSkewSummary;
};

}  // namespace icts
