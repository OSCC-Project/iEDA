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
#include "ViaResistanceModel.hh"

#include <algorithm>
#include <utility>

#include "Geometry.hh"
#include "ProcessCorner.hpp"
#include "ResistanceTemperature.hh"
#include "TopoPool.hh"

namespace ircx {

namespace {

auto viaDrawLengthWidth(const TopoEdge& edge, Micron micron_per_dbu) -> std::pair<F64, F64>
{
  const F64 dx = static_cast<F64>(geom::delta_x(edge.shape())) * micron_per_dbu;
  const F64 dy = static_cast<F64>(geom::delta_y(edge.shape())) * micron_per_dbu;
  return {std::max(dx, dy), std::min(dx, dy)};
}

auto viaResistanceArea(const TopoEdge& edge,
                       const itf::ProcessCorner& corner,
                       const itf::LayerVia& layer,
                       Micron micron_per_dbu) -> F64
{
  auto [length, width] = viaDrawLengthWidth(edge, micron_per_dbu);
  const F64 half_node_scale_factor = corner.get_half_node_scale_factor();
  length *= half_node_scale_factor;
  width *= half_node_scale_factor;

  if (layer.use_etch_vwl_for_resistance()) {
    const auto etch = layer.get_etch_vwl().query_interpolation(static_cast<float>(width), static_cast<float>(length));
    if (etch) {
      const F64 length_etch = etch->first;
      const F64 width_etch = etch->second;
      length = std::max<F64>(0.0, length - 2.0 * length_etch);
      width = std::max<F64>(0.0, width - 2.0 * width_etch);
    }
  }

  return length * width;
}

}  // namespace

auto ViaResistanceModel::calc(const TopoEdge& edge, const itf::ProcessCorner& corner, const itf::LayerVia& layer, Micron micron_per_dbu,
                              F64 operating_temperature) -> F64
{
  const F64 via_area = viaResistanceArea(edge, corner, layer, micron_per_dbu);

  F64 via_resistance = 0.0;
  if (auto rpv = layer.get_rpv()) {
    via_resistance = rpv.value();
  } else {
    via_resistance = layer.query_rpv_vs_area(via_area);
  }

  const ResistanceTemperatureCoefficients coefficients =
      resistanceTemperatureCoefficients(layer, [&](auto& crt1, auto& crt2) {
        layer.query_crt_vs_area(via_area, crt1, crt2);
      });
  return applyResistanceTemperatureDerating(via_resistance, operating_temperature, resistanceNominalTemperature(layer, corner),
                                            coefficients);
}

}  // namespace ircx
