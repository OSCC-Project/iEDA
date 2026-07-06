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
#include <span>
#include <vector>

#include "Types.hh"
#include "TopoPool.hh"
#include "NetEnvironment.hh"
#include "log/Log.hh"
namespace ircx {

class TrackOverlapMerge
{
 public:
  void compute(Dbu query_a0,
               Dbu query_a1,
               const std::vector<TrackOverlap>& dn_in,
               const std::vector<TrackOverlap>& up_in,
               std::vector<EdgeEnvironmentInterval>& out) const
  {
    if (query_a0 > query_a1) {
      std::swap(query_a0, query_a1);
    }

    out.clear();
    if (!(query_a0 < query_a1)) {
      return;
    }

    std::vector<TrackOverlap> dn;
    std::vector<TrackOverlap> up;

    normalizeSide(query_a0, query_a1, dn_in, dn);
    normalizeSide(query_a0, query_a1, up_in, up);

    mergeTwoSides(dn, up, out);
  }

 private:
  static TrackOverlap makeNullOverlap(Dbu a0, Dbu a1)
  {
    TrackOverlap ov;
    ov.a0 = a0;
    ov.a1 = a1;
    ov.sp = kMaxDbu;
    ov.edge = nullptr;
    return ov;
  }

  static void emitTrackOverlap(std::vector<TrackOverlap>& out, const TrackOverlap& ov)
  {
    if (!(ov.a0 < ov.a1)) {
      return;
    }

    if (!out.empty() &&
        out.back().a1 == ov.a0 &&
        out.back().edge == ov.edge &&
        out.back().sp == ov.sp) {
      out.back().a1 = ov.a1;
      return;
    }

    out.push_back(ov);
  }

  void normalizeSide(Dbu query_a0,
                     Dbu query_a1,
                     const std::vector<TrackOverlap>& in,
                     std::vector<TrackOverlap>& out) const
  {
    out.clear();

    std::vector<TrackOverlap> tmp;
    tmp.reserve(in.size());

    for (const auto& ov : in) {
      const Dbu a0 = std::max(query_a0, ov.a0);
      const Dbu a1 = std::min(query_a1, ov.a1);
      if (!(a0 < a1)) {
        continue;
      }

      TrackOverlap clipped = ov;
      clipped.a0 = a0;
      clipped.a1 = a1;
      tmp.push_back(clipped);
    }

    std::sort(tmp.begin(), tmp.end(), [](const TrackOverlap& lhs, const TrackOverlap& rhs) {
      if (lhs.a0 != rhs.a0) {
        return lhs.a0 < rhs.a0;
      }
      if (lhs.a1 != rhs.a1) {
        return lhs.a1 < rhs.a1;
      }
      if (lhs.edge != rhs.edge) {
        return lhs.edge < rhs.edge;
      }
      return lhs.sp < rhs.sp;
    });

    Dbu cursor = query_a0;
    for (const auto& ov : tmp) {
      LOG_FATAL_IF(ov.a0 < cursor)
          << "TrackOverlapMerge: same-side overlaps are not allowed. "
          << "prev_end=" << cursor << ", cur=[" << ov.a0 << ", " << ov.a1 << ")";

      if (cursor < ov.a0) {
        emitTrackOverlap(out, makeNullOverlap(cursor, ov.a0));
      }

      emitTrackOverlap(out, ov);
      cursor = ov.a1;
    }

    if (cursor < query_a1) {
      emitTrackOverlap(out, makeNullOverlap(cursor, query_a1));
    }

    if (out.empty()) {
      out.push_back(makeNullOverlap(query_a0, query_a1));
    }
  }

  static void emitOutput(std::vector<EdgeEnvironmentInterval>& out,
                         Dbu a0,
                         Dbu a1,
                         const TrackOverlap& dn,
                         const TrackOverlap& up)
  {
    if (!(a0 < a1)) {
      return;
    }

    const Dbu l_sp = dn.sp;
    const Dbu h_sp = up.sp;
    const TopoEdge* l_edge = dn.edge;
    const TopoEdge* h_edge = up.edge;

    if (!out.empty() &&
        out.back().a1 == a0 &&
        out.back().lo_adjacent == l_edge &&
        out.back().hi_adjacent == h_edge &&
        out.back().lo_spacing == l_sp &&
        out.back().hi_spacing == h_sp) {
      out.back().a1 = a1;
      return;
    }

    EdgeEnvironmentInterval iv;
    iv.a0 = a0;
    iv.a1 = a1;
    iv.lo_adjacent = l_edge;
    iv.hi_adjacent = h_edge;
    iv.lo_spacing = l_sp;
    iv.hi_spacing = h_sp;
    out.push_back(iv);
  }

  static void mergeTwoSides(const std::vector<TrackOverlap>& dn,
                            const std::vector<TrackOverlap>& up,
                            std::vector<EdgeEnvironmentInterval>& out)
  {
    out.clear();
    if (dn.empty() || up.empty()) {
      return;
    }

    Size i = 0;
    Size j = 0;

    while (i < dn.size() && j < up.size()) {
      const Dbu s = std::max(dn[i].a0, up[j].a0);
      const Dbu t = std::min(dn[i].a1, up[j].a1);

      if (s < t) {
        emitOutput(out, s, t, dn[i], up[j]);
      }

      if (dn[i].a1 == t) {
        ++i;
      }
      if (up[j].a1 == t) {
        ++j;
      }
    }
  }
};

class PixelOverlapMerge
{
 public:
  struct LayerPixelOverlaps {
    Size layer{0};                  // 0 means invalid / absent
    std::vector<PixelOverlap> segs; // higher priority comes first in input order
  };

  void compute(Dbu query_a0,
               Dbu query_a1,
               const std::vector<LayerPixelOverlaps>& dn_inputs,
               const std::vector<LayerPixelOverlaps>& up_inputs,
               std::vector<CrossOverlapSub>& out) const
  {
    if (query_a0 > query_a1) {
      std::swap(query_a0, query_a1);
    }

    out.clear();
    if (!(query_a0 < query_a1)) {
      return;
    }

    std::vector<NormalizedLayerPixelOverlaps> dn_norm;
    std::vector<NormalizedLayerPixelOverlaps> up_norm;
    dn_norm.reserve(dn_inputs.size());
    up_norm.reserve(up_inputs.size());

    for (const auto& in : dn_inputs) {
      if (in.layer == 0) {
        continue;
      }
      NormalizedLayerPixelOverlaps ni;
      ni.layer = in.layer;
      normalizeOne(query_a0, query_a1, in.segs, ni.segs);
      if (!ni.segs.empty()) {
        dn_norm.push_back(std::move(ni));
      }
    }

    for (const auto& in : up_inputs) {
      if (in.layer == 0) {
        continue;
      }
      NormalizedLayerPixelOverlaps ni;
      ni.layer = in.layer;
      normalizeOne(query_a0, query_a1, in.segs, ni.segs);
      if (!ni.segs.empty()) {
        up_norm.push_back(std::move(ni));
      }
    }

    std::vector<Dbu> bp;
    bp.reserve(2 + countBreakpoints(dn_norm) + countBreakpoints(up_norm));
    bp.push_back(query_a0);
    bp.push_back(query_a1);

    addBreakpoints(dn_norm, bp);
    addBreakpoints(up_norm, bp);

    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end()), bp.end());

    if (bp.size() < 2) {
      return;
    }

    std::vector<Size> dn_cursor(dn_norm.size(), 0);
    std::vector<Size> up_cursor(up_norm.size(), 0);

    for (Size k = 0; k + 1 < bp.size(); ++k) {
      const Dbu a0 = bp[k];
      const Dbu a1 = bp[k + 1];
      if (!(a0 < a1)) {
        continue;
      }

      const Size blw_layer = firstCoveringLayer(dn_norm, dn_cursor, a0, a1);
      const Size abv_layer = firstCoveringLayer(up_norm, up_cursor, a0, a1);

      emit(a0, a1, blw_layer, abv_layer, out);
    }
  }

 private:
  struct NormalizedLayerPixelOverlaps {
    Size layer{0};
    std::vector<PixelOverlap> segs;
  };

  static Size countBreakpoints(const std::vector<NormalizedLayerPixelOverlaps>& inputs)
  {
    Size n = 0;
    for (const auto& in : inputs) {
      n += static_cast<Size>(in.segs.size() * 2);
    }
    return n;
  }

  static void addBreakpoints(const std::vector<NormalizedLayerPixelOverlaps>& inputs,
                             std::vector<Dbu>& bp)
  {
    for (const auto& in : inputs) {
      for (const auto& seg : in.segs) {
        bp.push_back(seg.a0);
        bp.push_back(seg.a1);
      }
    }
  }

  static void normalizeOne(Dbu query_a0,
                           Dbu query_a1,
                           const std::vector<PixelOverlap>& in,
                           std::vector<PixelOverlap>& out)
  {
    out.clear();
    out.reserve(in.size());

    for (const auto& ov : in) {
      const Dbu a0 = std::max(query_a0, ov.a0);
      const Dbu a1 = std::min(query_a1, ov.a1);
      if (a0 < a1) {
        out.push_back(PixelOverlap{a0, a1});
      }
    }

    std::sort(out.begin(), out.end(), [](const PixelOverlap& lhs, const PixelOverlap& rhs) {
      if (lhs.a0 != rhs.a0) {
        return lhs.a0 < rhs.a0;
      }
      return lhs.a1 < rhs.a1;
    });

    std::vector<PixelOverlap> merged;
    merged.reserve(out.size());

    for (const auto& ov : out) {
      if (merged.empty() || merged.back().a1 < ov.a0) {
        merged.push_back(ov);
      } else {
        merged.back().a1 = std::max(merged.back().a1, ov.a1);
      }
    }

    out.swap(merged);
  }

  static void advanceCursor(const std::vector<PixelOverlap>& segs, Size& idx, Dbu x)
  {
    while (idx < segs.size() && segs[idx].a1 <= x) {
      ++idx;
    }
  }

  static bool covers(const std::vector<PixelOverlap>& segs, Size idx, Dbu a0, Dbu a1)
  {
    return idx < segs.size() && segs[idx].a0 < a1 && segs[idx].a1 > a0;
  }

  static Size firstCoveringLayer(const std::vector<NormalizedLayerPixelOverlaps>& inputs,
                                 std::vector<Size>& cursors,
                                 Dbu a0,
                                 Dbu a1)
  {
    for (Size i = 0; i < inputs.size(); ++i) {
      advanceCursor(inputs[i].segs, cursors[i], a0);
      if (covers(inputs[i].segs, cursors[i], a0, a1)) {
        return inputs[i].layer;
      }
    }
    return Size{0};
  }

  static void emit(Dbu a0,
                   Dbu a1,
                   Size blw_layer,
                   Size abv_layer,
                   std::vector<CrossOverlapSub>& out)
  {
    if (!(a0 < a1)) {
      return;
    }

    // The uncovered span is represented implicitly by gaps between emitted
    // cross-over segments and handled by downstream consumers as substrate/none.
    if (blw_layer == 0 && abv_layer == 0) {
      return;
    }

    if (!out.empty() &&
        out.back().a1 == a0 &&
        out.back().blw_layer == blw_layer &&
        out.back().abv_layer == abv_layer) {
      out.back().a1 = a1;
      return;
    }

    CrossOverlapSub sub;
    sub.a0 = a0;
    sub.a1 = a1;
    sub.blw_layer = blw_layer;
    sub.abv_layer = abv_layer;
    out.push_back(sub);
  }
};

}  // namespace ircx
