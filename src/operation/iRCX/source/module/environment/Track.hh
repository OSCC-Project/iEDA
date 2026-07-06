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
#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <set>
#include <vector>

#include "IntervalUtils.hh"
#include "Types.hh"
#include "TopoPool.hh"
#include "log/Log.hh"
namespace ircx {

// Parallel Overlap
struct TrackOverlap {
  Dbu a0{0};
  Dbu a1{0};

  Dbu sp{kMaxDbu}; // unsigned spacing = |current_fixed - coord|
  // edge == nullptr means this interval is not covered by any matched edge.
  const TopoEdge* edge{nullptr};
};

struct OverlapWidenContext {
  Dbu track_distance;  // |current_track_idx - base_track_idx|
  Dbu overlap_len;     // current raw overlap length before widening
  const TopoEdge* edge;  // matched edge
};

class Track
{
 public:
  using OverlapWidenFunc = std::function<Dbu(const OverlapWidenContext&)>;

 private:
  struct RemainingInterval {
    Dbu a0{0};
    Dbu a1{0};
  };

  struct SearchContext {
    Dbu coord{0};
    Dbu base_track_idx{0};
    Dbu query_a0{0};
    Dbu query_a1{0};
    int step{0};
    const OverlapWidenFunc* widen_func{nullptr};
  };

  struct TopoEdgeFixedLess
  {
    using is_transparent = void;

    bool operator()(const TopoEdge* lhs, const TopoEdge* rhs) const
    {
      if (lhs == rhs) {
        return false;
      }
      if (lhs == nullptr || rhs == nullptr) {
        return lhs < rhs;
      }
      if (lhs->coord() != rhs->coord()) {
        return lhs->coord() < rhs->coord();
      }
      return lhs < rhs;
    }

    bool operator()(const TopoEdge* lhs, Dbu rhs_coord) const
    {
      return lhs->coord() < rhs_coord;
    }

    bool operator()(Dbu lhs_coord, const TopoEdge* rhs) const
    {
      return lhs_coord < rhs->coord();
    }
  };

  using EdgeSet = std::set<const TopoEdge*, TopoEdgeFixedLess>;

 public:
  Track() = default;
  ~Track() = default;

  // getter
  Dbu track_ori() const { return track_ori_; }
  Dbu track_num() const { return track_num_; }
  Dbu track_dlt() const { return track_dlt_; }
  Dbu bucket_ori() const { return bucket_ori_; }
  Dbu bucket_num() const { return bucket_num_; }
  Dbu bucket_dlt() const { return bucket_dlt_; }

  // setter
  void set_track_ori(Dbu v) { track_ori_ = v; }
  void set_track_num(Dbu v) { track_num_ = v; }
  void set_track_dlt(Dbu v) { track_dlt_ = v; }
  void set_bucket_ori(Dbu v) { bucket_ori_ = v; }
  void set_bucket_num(Dbu v) { bucket_num_ = v; }
  void set_bucket_dlt(Dbu v) { bucket_dlt_ = v; }

  // coordinate mapping
  Dbu coordToTrack(Dbu coord) const { return (coord - track_ori_) / track_dlt_; }
  Dbu coordToBucket(Dbu coord) const { return (coord - bucket_ori_) / bucket_dlt_; }

  bool initTrack()
  {
    if (track_num_ <= 0 || track_dlt_ <= 0) {
      LOG_ERROR << "Track parameters are not initialized!";
      return false;
    }
    if (bucket_num_ <= 0 || bucket_dlt_ <= 0) {
      LOG_ERROR << "Bucket parameters are not initialized!";
      return false;
    }

    track_buckets_.assign(track_num_, std::vector<EdgeSet>(bucket_num_));
    return true;
  }

  void addEdge(const TopoEdge& edge)
  {
    Dbu a0 = edge.lo();
    Dbu a1 = edge.hi();
    ircx::interval::normalize(a0, a1);

    const Dbu track_idx = coordToTrack(edge.coord());
    const Dbu bucket_idx0 = coordToBucket(a0);
    const Dbu bucket_idx1 = coordToBucket(a1 - 1);

    if (!trackValid(track_idx) || !bucketValid(bucket_idx0) || !bucketValid(bucket_idx1)) {
      LOG_ERROR << "The track/bucket index is out of range!";
      return;
    }

    for (Dbu b = bucket_idx0; b <= bucket_idx1; ++b) {
      track_buckets_[track_idx][b].insert(&edge);
    }
  }

  // search_track_num > 0:
  //   search upward for search_track_num tracks, including the track containing coord;
  //   every returned non-null edge must satisfy edge->coord() > coord
  //
  // search_track_num < 0:
  //   search downward for |search_track_num| tracks, including the track containing coord;
  //   every returned non-null edge must satisfy edge->coord() < coord
  //
  // interval semantics: open interval (a0, a1)
  //
  // return semantics:
  //   - edge != nullptr: this interval is covered by the returned edge
  //   - edge == nullptr: this interval remains uncovered after searching all requested tracks
  //
  // widen_func semantics:
  //   - input:
  //       OverlapWidenContext{
  //         track_distance = |current_track_idx - base_track_idx|,
  //         overlap_len    = current raw overlap length,
  //         edge           = matched edge
  //       }
  //   - output:
  //       single-side widening length
  //   - widening is symmetric:
  //       widened interval = (ov.a0 - ext, ov.a1 + ext)
  //   - widened interval is clipped into the original query interval [a0, a1]
  //
  // search strategy:
  //   1) on each track, collect candidate edges once from the initial remaining intervals;
  //   2) traverse those candidates in search direction order (nearest first by fixed);
  //   3) for each edge, compute overlap, widen it with widen_func, then clip to query interval;
  //   4) cut away the widened covered part from the remaining intervals;
  //   5) continue until no remaining interval is left or all candidates are consumed;
  //   6) recurse to the next track, until all requested tracks are processed or no remaining
  //      interval is left;
  //   7) any remaining uncovered intervals are returned with edge == nullptr.
  std::vector<TrackOverlap> get_overlap(const LineSegmentI& line_seg,
                                        Dbu search_track_num,
                                        const OverlapWidenFunc& widen_func = {}) const
  {
    std::vector<TrackOverlap> result;
    if (search_track_num == 0) {
      return result;
    }

    Dbu a0 = line_seg.lo;
    Dbu a1 = line_seg.hi;
    ircx::interval::normalize(a0, a1);
    if (!intervalValid({a0, a1})) {
      return result;
    }

    const Dbu coord = line_seg.coord;
    const Dbu query_a0 = a0;
    const Dbu query_a1 = a1;

    const Dbu base_track_idx = coordToTrack(coord);
    if (!trackValid(base_track_idx)) {
      LOG_ERROR << "The base track index is out of range!";
      return result;
    }

    std::vector<RemainingInterval> remaining;
    remaining.push_back({query_a0, query_a1});

    const int step = (search_track_num > 0) ? 1 : -1;
    const Dbu tracks_to_search = (search_track_num > 0) ? search_track_num : -search_track_num;

    SearchContext ctx;
    ctx.coord = coord;
    ctx.base_track_idx = base_track_idx;
    ctx.query_a0 = query_a0;
    ctx.query_a1 = query_a1;
    ctx.step = step;
    ctx.widen_func = &widen_func;

    remaining = searchAcrossTracks(base_track_idx, tracks_to_search, remaining, result, ctx);

    for (const auto& iv : remaining) {
      TrackOverlap ov;
      ov.a0 = iv.a0;
      ov.a1 = iv.a1;
      ov.sp = kMaxDbu;
      ov.edge = nullptr;
      result.push_back(ov);
    }

    return result;
  }

 private:
  static bool intervalValid(const RemainingInterval& iv) { return iv.a0 < iv.a1; }

  static bool edgeIsInSearchDirection(const TopoEdge* edge, const SearchContext& ctx)
  {
    if (edge == nullptr) {
      return false;
    }
    return (ctx.step > 0) ? (edge->coord() > ctx.coord) : (edge->coord() < ctx.coord);
  }

  static TrackOverlap applyWidenAndClip(const TrackOverlap& ov,
                                        Dbu track_idx,
                                        const SearchContext& ctx)
  {
    TrackOverlap widened = ov;

    if (ctx.widen_func != nullptr && *ctx.widen_func && ov.edge != nullptr) {
      const OverlapWidenContext widen_ctx{
          .track_distance = std::abs(track_idx - ctx.base_track_idx),
          .overlap_len = ov.a1 - ov.a0,
          .edge = ov.edge,
      };

      Dbu ext = (*ctx.widen_func)(widen_ctx);
      if (ext < Dbu{0}) {
        ext = Dbu{0};
      }

      widened.a0 -= ext;
      widened.a1 += ext;
    }

    // clip into original query interval [query_a0, query_a1]
    widened.a0 = std::clamp(widened.a0, ctx.query_a0, ctx.query_a1);
    widened.a1 = std::clamp(widened.a1, ctx.query_a0, ctx.query_a1);

    return widened;
  }

  EdgeSet collectCandidateEdgesOnTrack(Dbu track_idx,
                                       const std::vector<RemainingInterval>& remaining) const
  {
    EdgeSet ordered;
    if (!trackValid(track_idx)) {
      return ordered;
    }

    for (const auto& iv : remaining) {
      Dbu bucket_idx0 = coordToBucket(iv.a0);
      Dbu bucket_idx1 = coordToBucket(iv.a1 - 1);
      if (bucket_idx0 > bucket_idx1) {
        std::swap(bucket_idx0, bucket_idx1);
      }

      if (!bucketValid(bucket_idx0) || !bucketValid(bucket_idx1)) {
        continue;
      }

      for (Dbu b = bucket_idx0; b <= bucket_idx1; ++b) {
        const auto& edge_set = track_buckets_[track_idx][b];
        ordered.insert(edge_set.begin(), edge_set.end());
      }
    }

    return ordered;
  }

  bool edgeHitsRemaining(const TopoEdge* edge, const std::vector<RemainingInterval>& remaining) const
  {
    if (edge == nullptr) {
      return false;
    }

    Dbu edge_a0 = edge->lo();
    Dbu edge_a1 = edge->hi();
    ircx::interval::normalize(edge_a0, edge_a1);

    for (const auto& iv : remaining) {
      if (ircx::interval::overlaps(edge_a0, edge_a1, iv.a0, iv.a1)) {
        return true;
      }
    }
    return false;
  }

  std::vector<TrackOverlap> computeEdgeOverlaps(Dbu track_idx,
                                                const TopoEdge* edge,
                                                const std::vector<RemainingInterval>& remaining,
                                                const SearchContext& ctx) const
  {
    std::vector<TrackOverlap> overlaps;
    if (edge == nullptr) {
      return overlaps;
    }

    Dbu edge_a0 = edge->lo();
    Dbu edge_a1 = edge->hi();
    ircx::interval::normalize(edge_a0, edge_a1);

    for (const auto& iv : remaining) {
      if (!ircx::interval::overlaps(edge_a0, edge_a1, iv.a0, iv.a1)) {
        continue;
      }

      const auto overlap = ircx::interval::intersection(edge_a0, edge_a1, iv.a0, iv.a1);
      TrackOverlap ov;
      ov.a0 = overlap.a0;
      ov.a1 = overlap.a1;
      ov.sp = std::abs(edge->coord() - ctx.coord);
      ov.edge = edge;

      if (ov.a0 < ov.a1) {
        TrackOverlap widened = applyWidenAndClip(ov, track_idx, ctx);

        // widened interval should not cross the current remaining fragment
        widened.a0 = std::max(widened.a0, iv.a0);
        widened.a1 = std::min(widened.a1, iv.a1);

        if (widened.a0 < widened.a1) {
          overlaps.push_back(widened);
        }
      }
    }

    return overlaps;
  }

  // Iteratively consume one track using a single ordered candidate set.
  // The key invariant is that `remaining` only shrinks, so an edge that does not
  // belong to the initial candidate set can never become relevant later.
  void searchWithinTrack(Dbu track_idx,
                        std::vector<RemainingInterval> remaining,
                        std::vector<TrackOverlap>& result,
                        const SearchContext& ctx) const
  {
    if (!trackValid(track_idx) || remaining.empty()) {
      return;
    }

    const EdgeSet ordered = collectCandidateEdgesOnTrack(track_idx, remaining);
    if (ordered.empty()) {
      return;
    }

    auto consume_edge = [&](const TopoEdge* edge) {
      if (!edgeHitsRemaining(edge, remaining)) {
        return;
      }

      const auto overlaps = computeEdgeOverlaps(track_idx, edge, remaining, ctx);
      if (overlaps.empty()) {
        return;
      }

      result.insert(result.end(), overlaps.begin(), overlaps.end());

      auto next_remaining = remaining;
      for (const auto& ov : overlaps) {
        next_remaining = ircx::interval::subtract(next_remaining, ov.a0, ov.a1);
        if (next_remaining.empty()) {
          break;
        }
      }

      remaining = std::move(next_remaining);
    };

    if (ctx.step > 0) {
      for (auto it = ordered.upper_bound(ctx.coord);
           it != ordered.end() && !remaining.empty();
           ++it) {
        consume_edge(*it);
      }
    } else {
      auto it = ordered.lower_bound(ctx.coord);
      for (auto rit = std::make_reverse_iterator(it);
           rit != ordered.rend() && !remaining.empty();
           ++rit) {
        consume_edge(*rit);
      }
    }
  }

  // Recursively process tracks in the requested direction.
  // For each track:
  //   1) consume as much remaining interval as possible on this track;
  //   2) commit only the widened+clipped overlap pieces to the final result;
  //   3) subtract those committed overlap pieces from remaining;
  //   4) recurse to the next track;
  //   5) return whatever interval is still uncovered after all requested tracks are processed.
  std::vector<RemainingInterval> searchAcrossTracks(Dbu track_idx,
                                              Dbu tracks_left,
                                              std::vector<RemainingInterval> remaining,
                                              std::vector<TrackOverlap>& result,
                                              const SearchContext& ctx) const
  {
    if (tracks_left <= 0 || remaining.empty()) {
      return remaining;
    }

    if (!trackValid(track_idx)) {
      return remaining;
    }

    std::vector<TrackOverlap> local_overlaps;
    searchWithinTrack(track_idx, remaining, local_overlaps, ctx);

    for (const auto& ov : local_overlaps) {
      // Defensive check: the directional constraint should already be guaranteed by
      if (!edgeIsInSearchDirection(ov.edge, ctx)) {
        continue;
      }

      result.push_back(ov);
      remaining = ircx::interval::subtract(remaining, ov.a0, ov.a1);
      if (remaining.empty()) {
        return remaining;
      }
    }

    return searchAcrossTracks(track_idx + ctx.step, tracks_left - 1, remaining, result, ctx);
  }

 private:
  // each track -> multiple buckets
  std::vector<std::vector<EdgeSet>> track_buckets_;

  Dbu track_ori_{0};
  Dbu track_num_{0};
  Dbu track_dlt_{0};

  Dbu bucket_ori_{0};
  Dbu bucket_num_{0};
  Dbu bucket_dlt_{10000};

  bool trackValid(Dbu t) const { return 0 <= t && t < track_num_; }
  bool bucketValid(Dbu b) const { return 0 <= b && b < bucket_num_; }
};

}  // namespace ircx
