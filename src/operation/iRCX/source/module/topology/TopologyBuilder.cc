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
#include "TopologyBuilder.hh"

#include <algorithm>
#include <omp.h>

#include "HashUtils.hh"
#include "LayoutData.hh"
namespace ircx {

// ---------------------------------------------------------------------------
// build_one_  (stateless – safe to call from multiple threads simultaneously)
// ---------------------------------------------------------------------------
TopologyBuilder::NetTopo TopologyBuilder::build_one_(const Net& net) const
{
  TopologyBuilder::NetTopo result;

  const Size net_id = net.id;

  auto& nodes = result.nodes;
  auto& edges = result.edges;
  std::vector<char> node_shape_valid;

  // Helpers that write only into this net's local vectors.
  auto append_node = [&](TopoNode node, bool shape_valid) -> Size {
    nodes.push_back(std::move(node));
    node_shape_valid.push_back(shape_valid);
    return nodes.size() - 1;
  };
  auto append_edge = [&](TopoEdge edge) -> Size {
    edges.push_back(std::move(edge));
    return edges.size() - 1;
  };
  auto merge_node_shape = [&](Size node_idx, const GtlRectI& rect) {
    TopoNode& node = nodes[node_idx];
    if (!node_shape_valid[node_idx]) {
      node.set_shape(rect);
      node_shape_valid[node_idx] = true;
      return;
    }

    const GtlRectI& old_rect = node.shape();
    node.set_shape(geom::make_rect<GtlRectI>(
        std::min(geom::min_x(old_rect), geom::min_x(rect)),
        std::min(geom::min_y(old_rect), geom::min_y(rect)),
        std::max(geom::max_x(old_rect), geom::max_x(rect)),
        std::max(geom::max_y(old_rect), geom::max_y(rect))));
  };
  auto endpoint_shape = [](const Segment& wire, const GtlPointI& point) -> GtlRectI {
    if (geom::is_horizontal_dominant(wire.p0, wire.p1)) {
      return geom::make_rect<GtlRectI>(
          geom::x(point), geom::min_y(wire.rect),
          geom::x(point), geom::max_y(wire.rect));
    }
    return geom::make_rect<GtlRectI>(
        geom::min_x(wire.rect), geom::y(point),
        geom::max_x(wire.rect), geom::y(point));
  };

  // (layer, point) -> local node index
  std::unordered_map<std::pair<Size, GtlPointI>,
                     Size,
                     hash::LayerPointHasher> local_node_index_by_key;

  // Track whether each pin has already been matched to a topology node.
  std::map<Str, bool> pin_consumed;
  for (const auto& pin : net.pins) {
    pin_consumed[pin.name] = false;
  }

  struct PinMatch
  {
    Str name;
    GtlRectI rect;
    bool matched{false};
  };

  // Match a (layer_id, point) against the remaining pin entries.
  // Returns the matched pin name and marks it consumed; returns empty if none matches.
  auto match_pin = [&](Size layer_id, GtlPointI point) -> PinMatch {
    for (const auto& pin : net.pins) {
      if (pin_consumed[pin.name])
        continue;

      for (const auto& [pin_layer_id, pin_rect] : pin.layer_id_rects) {
        if (layer_id != pin_layer_id)
          continue;
        if (geom::rect_contains_point(pin_rect, point)) {
          pin_consumed[pin.name] = true;
          return {pin.name, pin_rect, true};
        }
      }
    }
    return {};
  };

  // Single-pass node building: iterate segments then vias, deduplicating via
  // local_node_index_by_key. This replaces the original two-pass approach (separate
  // points-set collection followed by node construction).
  auto append_node_if_absent = [&](Size layer_id, GtlPointI point) {
    const auto node_key = std::make_pair(layer_id, point);
    if (local_node_index_by_key.count(node_key))
      return;

    TopoNode node(net_id);
    node.set_layer_id(layer_id);
    node.set_point(point);

    const PinMatch pin_match = match_pin(layer_id, point);
    bool shape_valid = false;
    if (pin_match.matched) {
      node.set_pin_name(pin_match.name);
      node.set_shape(pin_match.rect);
      shape_valid = true;
    } else {
      node.set_shape(geom::box_around(point, 1));
    }

    const Size node_idx = append_node(std::move(node), shape_valid);
    local_node_index_by_key[node_key] = node_idx;
  };

  for (const Segment& wire : net.segments) {
    const Size layer_id = wire.layer_id;
    append_node_if_absent(layer_id, wire.p0);
    append_node_if_absent(layer_id, wire.p1);
  }

  for (const Via& via : net.vias) {
    const GtlPointI via_point = via.point;
    append_node_if_absent(via.layer_rect_top.first, via_point);
    append_node_if_absent(via.layer_rect_btm.first, via_point);
  }

  // edges: wire — u_/v_ are LOCAL node indices here; remapped to global in addNet()
  for (const Segment& wire : net.segments) {
    const Size layer_id = wire.layer_id;
    const GtlPointI start_point = wire.p0;
    const GtlPointI end_point = wire.p1;

    const Size start_node_idx = local_node_index_by_key.at({layer_id, start_point});
    const Size end_node_idx = local_node_index_by_key.at({layer_id, end_point});

    merge_node_shape(start_node_idx, endpoint_shape(wire, start_point));
    merge_node_shape(end_node_idx, endpoint_shape(wire, end_point));

    TopoEdge edge(net_id);
    edge.set_layer_id(layer_id);
    if (geom::is_lower_left(start_point, end_point)) {
      edge.set_u(start_node_idx);
      edge.set_v(end_node_idx);
    } else {
      edge.set_u(end_node_idx);
      edge.set_v(start_node_idx);
    }

    edge.set_shape(wire.rect);
    append_edge(std::move(edge));
  }

  // edges: via — u_/v_ are LOCAL node indices here; remapped to global in addNet()
  for (const Via& via : net.vias) {
    const Size top_layer_id = via.layer_rect_top.first;
    const Size cut_layer_id = via.layer_rect_cut.first;
    const Size bottom_layer_id = via.layer_rect_btm.first;

    const GtlRectI cut_rect = via.layer_rect_cut.second;
    const GtlPointI via_point = via.point;

    const Size top_node_idx = local_node_index_by_key.at({top_layer_id, via_point});
    const Size bottom_node_idx = local_node_index_by_key.at({bottom_layer_id, via_point});

    merge_node_shape(top_node_idx, via.layer_rect_top.second);
    merge_node_shape(bottom_node_idx, via.layer_rect_btm.second);

    TopoEdge edge(net_id);
    edge.set_layer_id(cut_layer_id);
    edge.set_u(top_node_idx);
    edge.set_v(bottom_node_idx);
    edge.set_shape(cut_rect);
    edge.set_via_name(via.name);

    append_edge(std::move(edge));
  }

  return result;
}

// ---------------------------------------------------------------------------
// build_all  – two-phase parallel build
// ---------------------------------------------------------------------------
void TopologyBuilder::build_all(const LayoutData& ld) const
{
  if (!topo_pool_)
    return;

  const std::vector<Net>& regular_nets = ld.net_vec;
  const Size net_count = ld.regular_net_count();

  // Pre-allocate to avoid any shared-container writes during the parallel phase.
  std::vector<TopologyBuilder::NetTopo> net_topologies(net_count);

// Phase 1: Build each net's topology into independent local storage.
//   - No shared mutable state → threads operate completely independently.
//   - dynamic scheduling handles variable net sizes efficiently.
#pragma omp parallel for schedule(dynamic)
  for (Size net_idx = 0; net_idx < net_count; ++net_idx) {
    net_topologies[net_idx] = std::move(build_one_(regular_nets[net_idx]));
  }

  // Phase 1.5: Pre-reserve pools (eliminates all reallocations in Phase 2)
  {
    Size total_nodes = 0, total_edges = 0;
    for (const auto& net_topology : net_topologies) {
      total_nodes += net_topology.nodes.size();
      total_edges += net_topology.edges.size();
    }
    topo_pool_->reserve(net_count, total_nodes, total_edges);
  }

  // Phase 2: Serially merge all per-net results into the shared contiguous
  //   pools. This step is O(total_elements) memory moves – fast in practice –
  //   and preserves the cache-friendly layout expected by downstream code.

  for (auto& net_topology : net_topologies) {
    topo_pool_->addNet(std::move(net_topology.nodes), std::move(net_topology.edges));
  }

// Phase 3: in edge, local node id → global id
#pragma omp parallel for schedule(dynamic)
  for (Size net_idx = 0; net_idx < net_count; ++net_idx) {
    auto net_edges = topo_pool_->net_edges(net_idx);
    for (auto& net_edge : net_edges) {
      const Size local_u = net_edge.u();
      const Size local_v = net_edge.v();
      net_edge.set_u(topo_pool_->node_index(net_idx, local_u));
      net_edge.set_v(topo_pool_->node_index(net_idx, local_v));
    }
  }
}

// ---------------------------------------------------------------------------
// build_special
// ---------------------------------------------------------------------------
void TopologyBuilder::build_special(const LayoutData& ld) const
{
  if (!topo_pool_)
    return;

  const Net& special_net = ld.special_net;

  const Size segment_count = special_net.segments.size();
  const Size patch_count = special_net.patches.size();

  std::vector<TopoEdge> edges(segment_count + patch_count, TopoEdge(kSpecialNetId));

  auto set_edge_shape = [&](Size edge_idx, Size layer_id, const GtlRectI& rect) {
    edges[edge_idx].set_layer_id(layer_id);
    edges[edge_idx].set_shape(rect);
  };

#pragma omp parallel for schedule(dynamic)
  for (Size segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    const Segment& segment = special_net.segments[segment_idx];
    set_edge_shape(segment_idx, segment.layer_id, segment.rect);
  }

#pragma omp parallel for schedule(dynamic)
  for (Size patch_idx = 0; patch_idx < patch_count; ++patch_idx) {
    const Patch& patch = special_net.patches[patch_idx];
    set_edge_shape(patch_idx + segment_count, patch.layer_id, patch.rect);
  }

  topo_pool_->addSpecialEdges(std::move(edges));
}

}  // namespace ircx
