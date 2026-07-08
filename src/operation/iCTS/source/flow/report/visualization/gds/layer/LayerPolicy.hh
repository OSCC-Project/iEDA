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
 * @file LayerPolicy.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief CTS visualization semantic layer and palette policy.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ClockLayout.hh"
#include "report/visualization/gds/writer/GdsStream.hh"

namespace icts {
struct DrawingInst;
struct DrawingSegment;
}  // namespace icts

namespace icts::visualization {

class LayerPolicy
{
 public:
  LayerPolicy() = default;

  auto logicCellLayer() -> GdsLayerKey;
  auto instLayer(const DrawingInst& inst) -> std::optional<GdsLayerKey>;
  auto segmentLayer(const DrawingSegment& segment) -> GdsLayerKey;
  auto getLayerProperties() const -> std::vector<GdsLayerProperty>;

  enum class LayerKind
  {
    kLogicCell,
    kRegularSink,
    kMacroSink,
    kRootBuffer,
    kHTreeBuffer,
    kRouteSegment
  };

  struct LayerKey
  {
    LayerKind kind = LayerKind::kLogicCell;
    ClockLayoutMode view = ClockLayoutMode::kUnknown;
    ClockLayoutPhase synthesis_phase = ClockLayoutPhase::kUnknown;
    LayoutNetRole net_role = LayoutNetRole::kUnknown;
    SinkDomainKind sink_domain = SinkDomainKind::kUnknown;
    std::size_t clock_index = 0U;
    int topology_level = -1;
  };

  struct LayerDescriptor
  {
    LayerKey key;
    std::string display_name;
    std::string color;
  };

  struct RegisteredLayer
  {
    LayerKey key;
    GdsLayerProperty property;
  };

 private:
  auto getLayer(const LayerDescriptor& descriptor) -> GdsLayerKey;

  std::vector<RegisteredLayer> _layers;
  int16_t _next_layer = 1;
};

}  // namespace icts::visualization
