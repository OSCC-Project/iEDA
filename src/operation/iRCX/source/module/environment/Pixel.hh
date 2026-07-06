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
#include <vector>

#include "IntervalUtils.hh"
#include "TopoPool.hh"
#include "log/Log.hh"
namespace ircx {

// Perpendicular Overlap
struct PixelOverlap
{
  Dbu a0{0};
  Dbu a1{0};
  bool empty() const { return a1 <= a0; }
};

class Pixel
{
 public:
  Pixel() = default;
  ~Pixel() = default;

  // getter
  Dbu x0() const { return x0_; }
  Dbu y0() const { return y0_; }
  Dbu nx() const { return nx_; }
  Dbu ny() const { return ny_; }
  Dbu dx() const { return dx_; }
  Dbu dy() const { return dy_; }

  // setter
  void set_x0(Dbu x0) { x0_ = x0; }
  void set_y0(Dbu y0) { y0_ = y0; }
  void set_nx(Dbu nx) { nx_ = nx; }
  void set_ny(Dbu ny) { ny_ = ny; }
  void set_dx(Dbu dx) { dx_ = dx; }
  void set_dy(Dbu dy) { dy_ = dy; }

  // coordinate mapping
  Dbu coordToXIdx(Dbu coord) const { return (coord - x0_) / dx_; }
  Dbu coordToYIdx(Dbu coord) const { return (coord - y0_) / dy_; }

  Dbu idxToXCoord(Dbu idx) const { return x0_ + idx * dx_; }
  Dbu idxToYCoord(Dbu idx) const { return y0_ + idx * dy_; }

  bool initPixel()
  {
    if (nx_ <= 0 || ny_ <= 0 || dx_ <= 0 || dy_ <= 0) {
      LOG_ERROR << "Grid parameters are not initialized!";
      return false;
    }

    pixel_.assign(nx_, std::vector<bool>(ny_, false));
    return true;
  }

  void addEdge(const TopoEdge& edge)
  {
    if (pixel_.empty() || pixel_.front().empty()) {
      if (!initPixel()) {
        return;
      }
    }

    const GtlRectI& rect = edge.shape();

    Dbu x0 = geom::min_x(rect);
    Dbu y0 = geom::min_y(rect);
    Dbu x1 = geom::max_x(rect);
    Dbu y1 = geom::max_y(rect);

    if (x0 >= x1 || y0 >= y1) {
      return;
    }

    Dbu x_idx0 = coordToXIdx(x0);
    Dbu y_idx0 = coordToYIdx(y0);
    Dbu x_idx1 = coordToXIdx(x1);
    Dbu y_idx1 = coordToYIdx(y1);

    if (!xValid(x_idx0) || !yValid(y_idx0) || !xValid(x_idx1) || !yValid(y_idx1)) {
      LOG_ERROR << "The x/y index is out of range!";
      return;
    }

    for (Dbu i = x_idx0; i <= x_idx1; ++i) {
      for (Dbu j = y_idx0; j <= y_idx1; ++j) {
        pixel_[i][j] = true;
      }
    }
  }

  std::vector<PixelOverlap> get_overlap(const LineSegmentI& line_seg) const
  {
    std::vector<PixelOverlap> ret;
    if (pixel_.empty() || pixel_.front().empty()) {
      return ret;
    }

    bool is_horz = line_seg.is_horz;
    Dbu fixed = line_seg.coord;
    Dbu a0 = line_seg.lo;
    Dbu a1 = line_seg.hi;

    ircx::interval::normalize(a0, a1);

    if (is_horz) {
      const Dbu fixed_y_idx = coordToYIdx(fixed);
      if (!yValid(fixed_y_idx)) {
        return ret;
      }

      return collectConductorRuns(
        a0,
        a1,
        [&](Dbu coord) { return coordToXIdx(coord); },
        [&](Dbu idx) { return xValid(idx); },
        [&](Dbu idx) { return pixel_[idx][fixed_y_idx]; },
        [&](Dbu idx) { return idxToXCoord(idx); });
    } else {
      const Dbu fixed_x_idx = coordToXIdx(fixed);
      if (!xValid(fixed_x_idx)) {
        return ret;
      }

      return collectConductorRuns(
        a0,
        a1,
        [&](Dbu coord) { return coordToYIdx(coord); },
        [&](Dbu idx) { return yValid(idx); },
        [&](Dbu idx) { return pixel_[fixed_x_idx][idx]; },
        [&](Dbu idx) { return idxToYCoord(idx); });
    }
  }

 private:
  template <typename CoordToIdx,
            typename IdxValid,
            typename CellGetter,
            typename IdxToCoord>
  std::vector<PixelOverlap> collectConductorRuns(Dbu a0,
                                                 Dbu a1,
                                                 CoordToIdx coord_to_idx,
                                                 IdxValid idx_valid,
                                                 CellGetter cell_getter,
                                                 IdxToCoord idx_to_coord) const
  {
    std::vector<PixelOverlap> ret;
    if (a0 >= a1) return ret;

    const Dbu a0_idx = coord_to_idx(a0);
    const Dbu a1_idx = coord_to_idx(a1) + 1;
    if (a0_idx > a1_idx) return ret;

    auto clamp_sequence_bounds = [&](Dbu coord_lo, Dbu coord_hi) {
      PixelOverlap seq;
      coord_lo = std::max(coord_lo, a0);
      coord_hi = std::min(coord_hi, a1);
      seq.a0 = coord_lo;
      seq.a1 = coord_hi;
      return seq;
    };

    auto cell_or_empty = [&](Dbu idx) -> bool {
      return idx_valid(idx) ? cell_getter(idx) : false;
    };

    bool current_type = cell_or_empty(a0_idx);
    Dbu run_start = a0_idx;

    auto push_sequence = [&](Dbu start_idx, Dbu end_idx_exclusive) {
      if (end_idx_exclusive <= start_idx) {
        return;
      }

      // Each occupied idx is a sample on the pixel lattice. The effective
      // overlap span is bounded by the midpoints of neighboring samples.
      const Dbu lo = ircx::interval::midpoint(idx_to_coord(start_idx), idx_to_coord(start_idx + 1));
      const Dbu hi = ircx::interval::midpoint(idx_to_coord(end_idx_exclusive - 1), idx_to_coord(end_idx_exclusive));

      PixelOverlap seq = clamp_sequence_bounds(lo, hi);
      if (!seq.empty()) {
        ret.push_back(seq);
      }
    };

    for (Dbu idx = a0_idx + 1; idx <= a1_idx; ++idx) {
      const bool cell = cell_or_empty(idx);
      if (cell != current_type) {
        if (current_type) {
          push_sequence(run_start, idx);
        }

        run_start = idx;
        current_type = cell;
      }
    }

    if (current_type) {
      push_sequence(run_start, a1_idx + 1);
    }

    return ret;
  }

 private:
  std::vector<std::vector<bool>> pixel_;  // true: conductor, false: empty

  Dbu x0_{0}, y0_{0};
  Dbu nx_{0}, ny_{0};
  Dbu dx_{0}, dy_{0};

  bool xValid(Dbu x) const { return 0 <= x && x < nx_; }
  bool yValid(Dbu y) const { return 0 <= y && y < ny_; }
};

}  // namespace ircx
