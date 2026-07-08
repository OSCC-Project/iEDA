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
 * @file OptimizationPreparation.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Preparation contracts for CTS post-synthesis optimization.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "FastSta.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts {
class Clock;
class ClockLayout;
class Design;
class Wrapper;
}  // namespace icts

namespace icts::clock_sizing_optimization {

struct ClockSizingMasterQueryInput
{
  Wrapper* wrapper = nullptr;
  const std::vector<std::string>* buffer_cell_masters = nullptr;
};

auto BuildClockRouteGeometry(const ClockLayout& clock_layout, std::size_t clock_index) -> FastStaClockRouteGeometry;
auto CaptureGraphProfile(const FastSTA& fast_sta, FastStaClockId clock_id) -> ClockSizingRuntimeProfile;
auto CopyOuterProfile(ClockSizingRuntimeProfile& destination, const ClockSizingRuntimeProfile& source) -> void;
auto CollectClockSizingBufferMasters(const ClockSizingMasterQueryInput& input) -> std::vector<ClockSizingBufferMaster>;
auto BuildClockSizingRouteTrees(const Design& design, const std::vector<Clock*>& clocks) -> ClockSizingRouteTreeCache;
auto FindMasterInfo(const std::vector<ClockSizingBufferMaster>& master_infos, std::string_view cell_master)
    -> const ClockSizingBufferMaster*;
auto CollectClockSizingCapLimits(const FastSTA& fast_sta, FastStaClockId clock_id) -> std::vector<ClockSizingCapLimit>;
auto CollectClockSizingSlewLimits(const FastSTA& fast_sta, FastStaClockId clock_id) -> std::vector<ClockSizingSlewLimit>;
auto CollectClockSizingBuffers(const Design& design, const FastSTA& fast_sta, FastStaClockId clock_id,
                               const std::vector<ClockSizingBufferMaster>& master_infos) -> std::vector<ClockSizingBuffer>;
auto InjectRouteTrees(const Design& design, FastSTA& fast_sta, FastStaClockId clock_id, const Clock& clock,
                      const ClockSizingRouteTreeCache& route_tree_by_net) -> bool;

}  // namespace icts::clock_sizing_optimization
