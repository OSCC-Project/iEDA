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
#include "EdgeCapAccumulator.hh"

namespace ircx {

namespace {

auto classifyAdjacent(const TopoEdge* adjacent, Size net_idx) -> AdjacentKind
{
  if (adjacent == nullptr) {
    return AdjacentKind::kNone;
  }
  if (adjacent->net_id() == kSpecialNetId) {
    return AdjacentKind::kSpecialNet;
  }
  if (adjacent->net_id() == net_idx) {
    return AdjacentKind::kSameNet;
  }
  return AdjacentKind::kOtherNet;
}

}  // namespace

auto makeSideContext(Micron spacing,
                     const TopoEdge* adjacent,
                     Size net_idx) -> SideContext
{
  return SideContext{spacing, adjacent, classifyAdjacent(adjacent, net_idx)};
}

EdgeCapAccumulator::EdgeCapAccumulator(const CapTableQuery& cap_query,
                                       const TopoPool& topo_pool,
                                       RCTable& rc_table,
                                       std::span<F64> edge_ground_caps,
                                       Size corner_idx,
                                       Size net_idx,
                                       Size edge_idx,
                                       Size edge_global_id)
    : cap_query_(cap_query),
      topo_pool_(topo_pool),
      rc_table_(rc_table),
      edge_ground_caps_(edge_ground_caps),
      corner_idx_(corner_idx),
      net_idx_(net_idx),
      edge_idx_(edge_idx),
      edge_global_id_(edge_global_id)
{
}

void EdgeCapAccumulator::accumulateSpan(Micron span_length,
                                        const Str& below_layer,
                                        const Str& above_layer,
                                        const SideContext& low_side,
                                        const SideContext& high_side)
{
  if (span_length <= 0.0) {
    return;
  }

  if (low_side.occupied() && high_side.occupied()) {
    const parser::CapacitanceResult low_side_cap =
        cap_query_.nearCap(below_layer, above_layer, low_side.spacing);
    const parser::CapacitanceResult high_side_cap =
        cap_query_.nearCap(below_layer, above_layer, high_side.spacing);

    accumulateGround(low_side, span_length * low_side_cap.ground_cap);
    accumulateGround(high_side, span_length * high_side_cap.ground_cap);
    foldCoupling(low_side, span_length * low_side_cap.coupling_cap / 2.0);
    foldCoupling(high_side, span_length * high_side_cap.coupling_cap / 2.0);
    return;
  }

  if (low_side.occupied() || high_side.occupied()) {
    const SideContext& occupied_side = low_side.occupied() ? low_side : high_side;
    const parser::CapacitanceResult occupied_side_cap =
        cap_query_.nearCap(below_layer, above_layer, occupied_side.spacing);

    accumulateGround(occupied_side, 2.0 * span_length * occupied_side_cap.ground_cap);
    foldCoupling(occupied_side, span_length * occupied_side_cap.coupling_cap);
    return;
  }

  const parser::CapacitanceResult far_cap =
      cap_query_.farthestCap(below_layer, above_layer);
  edge_ground_caps_[edge_idx_] +=
      2.0 * span_length * (far_cap.ground_cap + far_cap.coupling_cap);
}

void EdgeCapAccumulator::accumulateGround(const SideContext& side,
                                          double ground_cap_ff)
{
  if (ground_cap_ff <= 0.0 || !side.occupied()) {
    return;
  }

  edge_ground_caps_[edge_idx_] += side.sameNet() ? ground_cap_ff / 2.0
                                                 : ground_cap_ff;
}

void EdgeCapAccumulator::foldCoupling(const SideContext& side,
                                      double coupling_cap_ff)
{
  if (coupling_cap_ff <= 0.0 || !side.occupied()) {
    return;
  }
  if (side.specialNet()) {
    edge_ground_caps_[edge_idx_] += coupling_cap_ff;
    return;
  }
  if (side.sameNet()) {
    return;
  }

  const Size adjacent_edge_global_id = topo_pool_.edge_index(*side.adjacent);
  rc_table_.append_net_ccap_entry(
      net_idx_,
      edge_global_id_,
      adjacent_edge_global_id,
      corner_idx_,
      static_cast<F32>(coupling_cap_ff));
}

}  // namespace ircx
