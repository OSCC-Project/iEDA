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
#include <algorithm>
#include <omp.h>
#include <vector>

#include "Environment.hh"
#include "IntervalEngine.hh"
#include "IntervalUtils.hh"
#include "LayoutData.hh"
#include "TopoPool.hh"
#include "log/Log.hh"
namespace ircx {

namespace {

struct Axis
{
  Dbu origin{0};
  Dbu count{0};
  Dbu step{0};
};

auto ceilDivPositive(Dbu value, Dbu divisor) -> Dbu
{
  if (value <= 0 || divisor <= 0) {
    return 0;
  }
  return static_cast<Dbu>((static_cast<I64>(value) + divisor - 1) / divisor);
}

auto coverAxis(Dbu origin, Dbu count, Dbu step, Dbu lo, Dbu hi) -> Axis
{
  if (step <= 0) {
    return {origin, count, step};
  }

  I64 axis_origin = origin;
  I64 axis_count = count;
  const I64 axis_step = step;

  if (axis_origin > lo) {
    const I64 shift = (axis_origin - lo + axis_step - 1) / axis_step;
    axis_origin -= shift * axis_step;
    axis_count += shift;
  }

  const I64 covered_hi = axis_origin + axis_step * axis_count;
  if (covered_hi <= hi) {
    axis_count += (static_cast<I64>(hi) - covered_hi) / axis_step + 1;
  }

  return {static_cast<Dbu>(axis_origin), static_cast<Dbu>(axis_count), step};
}

}  // namespace

void Environment::reset()
{
  layout_data_ = nullptr;
  topo_pool_ = nullptr;
  layer_to_pixel_prefer_dir_.clear();
  layer_to_pixel_nonprefer_dir_.clear();
  layer_to_track_.clear();
  layer_to_search_track_num_.clear();
}

bool Environment::buildTracks()
{
  if (!layout_data_) {
    LOG_ERROR << "build environment tracks failed: LayoutData not initialized.";
    return false;
  }
  if (!topo_pool_) {
    LOG_ERROR << "build environment tracks failed: TopoPool not initialized.";
    return false;
  }

  const std::map<Size, RoutingLayer>& routing_layers = layout_data_->routing_layers;

  const GtlRectI& rect = layout_data_->die_shape;
  Dbu die_x0 = geom::min_x(rect);
  Dbu die_y0 = geom::min_y(rect);
  Dbu die_x1 = geom::max_x(rect);
  Dbu die_y1 = geom::max_y(rect);
  Dbu die_dx = geom::delta_x(rect);
  Dbu die_dy = geom::delta_y(rect);

  Dbu bucket_dlt = static_cast<Dbu>(bucket_size_um_ * layout_data_->dbu_per_micron);

  layer_to_track_.clear();

  // init
  for (const auto& [lid, layer] : routing_layers) {
    bool is_horz = layer.is_prefer_horz();
    const RoutingLayer::TrackInfo& ti = layer.track_info();
    Track track;

    Dbu track_ori = is_horz ? ti.y0 : ti.x0;
    Dbu track_num = is_horz ? ti.ny : ti.nx;
    Dbu track_dlt = is_horz ? ti.dy : ti.dx;
    const Dbu axis_lo = is_horz ? die_y0 : die_x0;
    const Dbu axis_hi = is_horz ? die_y1 : die_x1;
    const auto track_axis = coverAxis(track_ori, track_num, track_dlt, axis_lo, axis_hi);

    track.set_track_ori(track_axis.origin);
    track.set_track_num(track_axis.count);
    track.set_track_dlt(track_axis.step);

    Dbu bucket_len = is_horz ? die_dx : die_dy;
    track.set_bucket_ori(is_horz ? die_x0 : die_y0);
    track.set_bucket_num(ceilDivPositive(bucket_len, bucket_dlt));
    track.set_bucket_dlt(bucket_dlt);

    if (!track.initTrack()) {
      LOG_ERROR << "build environment tracks failed on layer " << lid;
      return false;
    }
    layer_to_track_[lid] = std::move(track);
  }

  // build: regular edges
  const std::vector<TopoEdge>& edge_pool = topo_pool_->edge_pool();
  for (const TopoEdge& edge : edge_pool) {
    if (edge.is_via()) continue;

    Size lid = edge.layer_id();
    bool layer_is_horz = routing_layers.at(lid).is_prefer_horz();

    if (edge.is_horz() == layer_is_horz) {
      layer_to_track_.at(lid).addEdge(edge);
    }
  }

  // build: special-net edges (power/ground context)
  for (const TopoEdge& edge : topo_pool_->special_edge_pool()) {
    if (edge.is_via()) continue;

    Size lid = edge.layer_id();
    bool layer_is_horz = routing_layers.at(lid).is_prefer_horz();

    if (edge.is_horz() == layer_is_horz) {
      layer_to_track_.at(lid).addEdge(edge);
    }
  }

  return true;
}

bool Environment::buildPixels()
{
  if (!layout_data_) {
    LOG_ERROR << "build environment pixels failed: LayoutData not initialized.";
    return false;
  }
  if (!topo_pool_) {
    LOG_ERROR << "build environment pixels failed: TopoPool not initialized.";
    return false;
  }

  const std::map<Size, RoutingLayer>& routing_layers = layout_data_->routing_layers;

  const GtlRectI& rect = layout_data_->die_shape;
  Dbu die_x0 = geom::min_x(rect);
  Dbu die_y0 = geom::min_y(rect);
  Dbu die_x1 = geom::max_x(rect);
  Dbu die_y1 = geom::max_y(rect);

  layer_to_pixel_prefer_dir_.clear();
  layer_to_pixel_nonprefer_dir_.clear();

  // init
  for (const auto& [lid, layer] : routing_layers) {
    const RoutingLayer::TrackInfo& ti = layer.track_info();
    Pixel pixel;

    Dbu x0 = ti.x0;
    Dbu y0 = ti.y0;
    Dbu nx = ti.nx;
    Dbu ny = ti.ny;
    Dbu dx = ti.dx;
    Dbu dy = ti.dy;
    // if (layer.is_prefer_horz()) {
    //   dx = layer.layer_width();
    // } else {
    //   dy = layer.layer_width();
    // }

    const auto x_axis = coverAxis(x0, nx, dx, die_x0, die_x1);
    const auto y_axis = coverAxis(y0, ny, dy, die_y0, die_y1);

    pixel.set_x0(x_axis.origin);
    pixel.set_nx(x_axis.count);
    pixel.set_dx(x_axis.step);
    
    pixel.set_y0(y_axis.origin);
    pixel.set_ny(y_axis.count);
    pixel.set_dy(y_axis.step);

    if (!pixel.initPixel()) {
      LOG_ERROR << "build environment pixels failed on layer " << lid;
      return false;
    }
    layer_to_pixel_prefer_dir_[lid] = pixel;
    layer_to_pixel_nonprefer_dir_[lid] = std::move(pixel);
  }

  auto add_pixel_edge = [&](const TopoEdge& edge) {
    if (edge.is_via()) {
      return;
    }

    Size lid = edge.layer_id();
    bool layer_is_horz = routing_layers.at(lid).is_prefer_horz();

    if (edge.is_horz() == layer_is_horz) {
      layer_to_pixel_prefer_dir_.at(lid).addEdge(edge);
    } else {
      layer_to_pixel_nonprefer_dir_.at(lid).addEdge(edge);
    }
  };

  // build: regular edges
  for (const TopoEdge& edge : topo_pool_->edge_pool()) {
    add_pixel_edge(edge);
  }

  // build: special-net edges (power/ground context)
  for (const TopoEdge& edge : topo_pool_->special_edge_pool()) {
    add_pixel_edge(edge);
  }

  return true;
}

void Environment::buildSearchTrackNumMap()
{
  const std::map<Size, RoutingLayer>& routing_layers = layout_data_->routing_layers;

  layer_to_search_track_num_.clear();

  for (const auto& [lid, layer] : routing_layers) {
    // Dbu window_size = static_cast<Dbu>(window_size_um_ * layout_data_->dbu_per_micron);
    // layer_to_search_track_num_[lid] = window_size / layer_to_track_[lid].track_dlt();
    layer_to_search_track_num_[lid] = 10;
  }
}

bool Environment::buildNetEnvironments(std::vector<NetEnvironment>& net_environments)
{
  if (!layout_data_) {
    LOG_ERROR << "build environment failed: LayoutData not initialized.";
    return false;
  }
  if (!topo_pool_) {
    LOG_ERROR << "build environment failed: TopoPool not initialized.";
    return false;
  }

  if (!buildTracks() || !buildPixels()) {
    return false;
  }
  buildSearchTrackNumMap();

  Size net_num = layout_data_->regular_net_count();
  net_environments.clear();
  net_environments.resize(net_num);

  const std::map<Size, RoutingLayer>& routing_layers = layout_data_->routing_layers;
  const Size min_lid = routing_layers.empty() ? 0 : routing_layers.begin()->first;
  const Size max_lid = routing_layers.empty() ? 0 : routing_layers.rbegin()->first;

  auto widen_me = [](const LineSegmentI& seg, Dbu ext) {
    LineSegmentI out = seg;
    out.lo -= ext;
    out.hi += ext;
    return out;
  };

  auto clip_cross_segs = [](const std::vector<CrossOverlapSub>& full, Dbu a0, Dbu a1) {
    return ircx::interval::clip(
        full,
        a0,
        a1,
        [](const CrossOverlapSub& lhs, const CrossOverlapSub& rhs) {
          return lhs.blw_layer == rhs.blw_layer && lhs.abv_layer == rhs.abv_layer;
        });
  };

  TrackOverlapMerge track_merger;
  PixelOverlapMerge pixel_merger;

  auto collect_cross_side = [&](const LineSegmentI& full_seg, Size base_lid, bool search_up) {
    std::vector<PixelOverlapMerge::LayerPixelOverlaps> bufs;

    for (Size delta = 1; delta <= cross_layer_; ++delta) {
      Size cand_lid = 0;

      if (search_up) {
        if (base_lid > max_lid || max_lid - base_lid < delta) {
          break;
        }
        cand_lid = base_lid + delta;
      } else {
        if (base_lid < min_lid || base_lid - min_lid < delta) {
          break;
        }
        cand_lid = base_lid - delta;
      }

      auto it_layer = routing_layers.find(cand_lid);
      if (it_layer == routing_layers.end()) {
        continue;
      }

      // Cross-over only queries the conductor set orthogonal to full_seg.
      const auto& pixel_map =
          (it_layer->second.is_prefer_horz() != full_seg.is_horz)
              ? layer_to_pixel_prefer_dir_
              : layer_to_pixel_nonprefer_dir_;

      auto it_pixel = pixel_map.find(cand_lid);
      if (it_pixel == pixel_map.end()) {
        continue;
      }

      std::vector<PixelOverlap> segs = it_pixel->second.get_overlap(full_seg);
      if (segs.empty()) {
        continue;
      }

      // Smaller layer deltas have higher priority in PixelOverlapMerge.
      PixelOverlapMerge::LayerPixelOverlaps in;
      in.layer = cand_lid;
      in.segs = std::move(segs);
      bufs.push_back(std::move(in));
    }

    return bufs;
  };

  #pragma omp parallel for schedule(dynamic)
  for (Size nid = 0; nid < net_num; nid++) {
    NetEnvironment& environment = net_environments[nid];
    environment.clear();

    for (const TopoEdge& edge : topo_pool_->net_edges(nid)) {
      if (edge.is_via()) {
        environment.appendEdgeIntervals({});  // placeholder to keep index aligned with TopoPool
        continue;
      }

      const Size lid = edge.layer_id();
      const LineSegmentI query_seg = widen_me(edge.line_segment(), 0);

      std::vector<TrackOverlap> track_ov_up =
          layer_to_track_[lid].get_overlap(query_seg,  layer_to_search_track_num_[lid], nullptr);
      std::vector<TrackOverlap> track_ov_dn =
          layer_to_track_[lid].get_overlap(query_seg, -layer_to_search_track_num_[lid], nullptr);

      std::vector<EdgeEnvironmentInterval> out;
      track_merger.compute(query_seg.lo, query_seg.hi, track_ov_dn, track_ov_up, out);

      const auto dn_inputs = collect_cross_side(query_seg, lid, /*search_up=*/false);
      const auto up_inputs = collect_cross_side(query_seg, lid, /*search_up=*/true);

      std::vector<CrossOverlapSub> cross_full;
      pixel_merger.compute(query_seg.lo, query_seg.hi, dn_inputs, up_inputs, cross_full);

      for (EdgeEnvironmentInterval& interval : out) {
        interval.cross_segs = clip_cross_segs(cross_full, interval.a0, interval.a1);
      }

      environment.appendEdgeIntervals(std::move(out));
    }
  }

  return true;
}

} // namespace ircx
