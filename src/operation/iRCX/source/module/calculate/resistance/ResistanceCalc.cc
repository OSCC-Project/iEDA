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
#include "ResistanceCalc.hh"

#include <algorithm>
#include <stdexcept>

#include "LayerTable.hh"
#include "LayoutData.hh"
#include "TopoPool.hh"
#include "ProcessCorner.hpp"
#include "ViaResistanceModel.hh"
#include "WireResistanceModel.hh"
#include "log/Log.hh"
namespace ircx {

bool ResistanceCalc::ProcessLayerResolver::build(
    const LayerTable& layer_table,
    const itf::ProcessCorner& corner,
    const TopoPool& topo_pool,
    const Str& corner_name)
{
  conductors_.clear();
  vias_.clear();

  if (corner.get_layers() == nullptr) {
    LOG_ERROR << "calculate resistance failed: process layers missing for corner "
              << corner_name << ".";
    return false;
  }

  for (const TopoEdge& edge : topo_pool.edge_pool()) {
    const Size design_layer_id = edge.layer_id();

    Str design_layer_name;
    Str process_layer_name;
    Size process_layer_id = kMaxSize;
    try {
      design_layer_name = layer_table.design_name(design_layer_id);
      process_layer_id = layer_table.design_to_process_id(design_layer_id);
      process_layer_name = layer_table.process_name(process_layer_id);
    } catch (const std::out_of_range&) {
      LOG_ERROR << "calculate resistance failed: layer mapping missing, corner="
                << corner_name << ", design_layer_id=" << design_layer_id << ".";
      return false;
    }

    if (edge.is_via()) {
      if (vias_.contains(design_layer_id)) {
        continue;
      }

      const itf::LayerVia* via_layer =
          corner.get_layers()->find_via_layer(process_layer_id);
      if (via_layer == nullptr) {
        LOG_ERROR << "calculate resistance failed: via layer not found, corner="
                  << corner_name << ", design_layer_id=" << design_layer_id
                  << ", design_layer=" << design_layer_name
                  << ", process_layer_id=" << process_layer_id
                  << ", process_layer=" << process_layer_name << ".";
        return false;
      }
      vias_.emplace(design_layer_id, via_layer);
      continue;
    }

    if (conductors_.contains(design_layer_id)) {
      continue;
    }

    const itf::LayerConductor* conductor_layer =
        corner.get_layers()->find_conductor_layer(process_layer_id);
    if (conductor_layer == nullptr) {
      LOG_ERROR << "calculate resistance failed: conductor layer not found, corner="
                << corner_name << ", design_layer_id=" << design_layer_id
                << ", design_layer=" << design_layer_name
                << ", process_layer_id=" << process_layer_id
                << ", process_layer=" << process_layer_name << ".";
      return false;
    }
    conductors_.emplace(design_layer_id, conductor_layer);
  }

  return true;
}

const itf::LayerConductor* ResistanceCalc::ProcessLayerResolver::conductor(
    Size design_layer_id) const
{
  const auto it = conductors_.find(design_layer_id);
  return it == conductors_.end() ? nullptr : it->second;
}

const itf::LayerVia* ResistanceCalc::ProcessLayerResolver::via(
    Size design_layer_id) const
{
  const auto it = vias_.find(design_layer_id);
  return it == vias_.end() ? nullptr : it->second;
}

bool ResistanceCalc::validateInputs() const
{
  if (layout_data_ == nullptr) {
    LOG_ERROR << "calculate resistance failed: LayoutData not initialized.";
    return false;
  }
  if (topo_pool_ == nullptr) {
    LOG_ERROR << "calculate resistance failed: TopoPool not initialized.";
    return false;
  }
  if (layer_table_ == nullptr) {
    LOG_ERROR << "calculate resistance failed: LayerTable not initialized.";
    return false;
  }
  if (corner_net_etch_pools_ == nullptr) {
    LOG_ERROR << "calculate resistance failed: etch profiles not set.";
    return false;
  }
  if (rc_table_ == nullptr) {
    LOG_ERROR << "calculate resistance failed: RC table not set.";
    return false;
  }
  if (corner_data_ == nullptr || corner_data_->empty()) {
    LOG_ERROR << "calculate resistance failed: process corners not set.";
    return false;
  }
  for (const auto& corner : *corner_data_) {
    if (corner.process_corner == nullptr) {
      LOG_ERROR << "calculate resistance failed: null process corner "
                << corner.name << ".";
      return false;
    }
  }
  if (corner_net_etch_pools_->corner_num() != corner_data_->size()
      || corner_net_etch_pools_->net_num() != layout_data_->regular_net_count()) {
    LOG_ERROR << "calculate resistance failed: etch profile dimensions mismatch.";
    return false;
  }

  return true;
}

bool ResistanceCalc::buildCornerViews(std::vector<CornerCalcView>& views) const
{
  views.clear();
  views.reserve(corner_data_->size());

  for (Size corner_idx = 0; corner_idx < corner_data_->size(); ++corner_idx) {
    const auto& corner_data = (*corner_data_)[corner_idx];

    CornerCalcView view;
    view.idx = corner_idx;
    view.data = &corner_data;
    view.process_corner = corner_data.process_corner.get();
    view.temperature = corner_data.temperature;
    if (!view.layers.build(*layer_table_, *view.process_corner,
                           *topo_pool_, corner_data.name)) {
      return false;
    }

    views.push_back(std::move(view));
  }

  return true;
}

bool ResistanceCalc::calc()
{
  if (!validateInputs()) {
    return false;
  }

  std::vector<CornerCalcView> corner_views;
  if (!buildCornerViews(corner_views)) {
    return false;
  }

  for (const CornerCalcView& corner : corner_views) {
    calcCorner(corner);
  }

  return true;
}

void ResistanceCalc::calcCorner(const CornerCalcView& corner) const
{
  const Size regular_net_count = layout_data_->regular_net_count();

  #pragma omp parallel for schedule(dynamic)
  for (Size net_idx = 0; net_idx < regular_net_count; ++net_idx) {
    calcNet(corner, net_idx);
  }
}

void ResistanceCalc::calcNet(const CornerCalcView& corner, Size net_idx) const
{
  const CornerNetId corner_net_id{corner.idx, net_idx};
  const auto net_edges = topo_pool_->net_edges(net_idx);
  auto edge_resistances = rc_table_->corner_net_res_pool(corner_net_id);
  const NetEtchProfile& etch_profile = corner_net_etch_pools_->at(corner_net_id);

  for (Size edge_idx = 0; edge_idx < net_edges.size(); ++edge_idx) {
    edge_resistances[edge_idx] =
        calcEdgeResistance(corner, net_idx, edge_idx, net_edges[edge_idx], etch_profile);
  }
}

F64 ResistanceCalc::calcEdgeResistance(const CornerCalcView& corner,
                                       Size net_idx,
                                       Size edge_idx,
                                       const TopoEdge& edge,
                                       const NetEtchProfile& etch_profile) const
{
  if (edge.is_via()) {
    return calcViaResistance(corner, edge);
  }

  const auto edge_etch_intervals = etch_profile.edgeIntervals(edge_idx);
  if (edge_etch_intervals.empty()) {
    LOG_ERROR << "calculate resistance warning: no etch intervals, corner="
              << corner.data->name << ", net_idx=" << net_idx
              << ", edge_idx=" << edge_idx << ".";
    return 0.0;
  }

  return calcConductorResistance(corner, edge, edge_etch_intervals);
}

F64 ResistanceCalc::calcViaResistance(const CornerCalcView& corner,
                                      const TopoEdge& edge) const
{
  const itf::LayerVia* via_layer = corner.layers.via(edge.layer_id());
  if (via_layer == nullptr) {
    LOG_ERROR << "calculate resistance failed: via layer not bound, corner="
              << corner.data->name << ", design_layer_id=" << edge.layer_id()
              << ".";
    return 0.0;
  }

  return ViaResistanceModel::calc(edge, *corner.process_corner, *via_layer,
                                  micron_per_dbu_, corner.temperature);
}

F64 ResistanceCalc::calcConductorResistance(
    const CornerCalcView& corner,
    const TopoEdge& edge,
    std::span<const EdgeEtchInterval> edge_etch_intervals) const
{
  const itf::LayerConductor* conductor_layer = corner.layers.conductor(edge.layer_id());
  if (conductor_layer == nullptr) {
    LOG_ERROR << "calculate resistance failed: conductor layer not bound, corner="
              << corner.data->name << ", design_layer_id=" << edge.layer_id()
              << ".";
    return 0.0;
  }

  return WireResistanceModel::calc(edgeSegment(edge), edge_etch_intervals,
                                   *corner.process_corner, *conductor_layer,
                                   corner.temperature);
}

LineSegment<Micron> ResistanceCalc::edgeSegment(const TopoEdge& edge) const
{
  const TopoNode& u_node = topo_pool_->node_at(edge.u());
  const TopoNode& v_node = topo_pool_->node_at(edge.v());

  LineSegment<Micron> segment;
  segment.is_horz = edge.is_horz();
  segment.coord = edge.coord() * micron_per_dbu_;
  if (edge.is_horz()) {
    segment.lo = geom::x(u_node.point()) * micron_per_dbu_;
    segment.hi = geom::x(v_node.point()) * micron_per_dbu_;
  } else {
    segment.lo = geom::y(u_node.point()) * micron_per_dbu_;
    segment.hi = geom::y(v_node.point()) * micron_per_dbu_;
  }
  if (segment.hi < segment.lo) {
    std::swap(segment.lo, segment.hi);
  }
  return segment;
}

}
