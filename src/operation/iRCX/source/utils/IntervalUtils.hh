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
#include <type_traits>
#include <utility>
#include <vector>

namespace ircx {
namespace interval {

// Ranges use non-zero-length interval semantics: [a0, a1) is valid only when
// a0 < a1, and two ranges that only touch at an endpoint do not overlap.
template <typename T>
struct Range {
  T a0{};
  T a1{};
};

template <typename T>
inline auto normalize(T& a0, T& a1) -> void
{
  if (a0 > a1) {
    std::swap(a0, a1);
  }
}

template <typename T>
inline auto is_valid(T a0, T a1) -> bool
{
  return a0 < a1;
}

template <typename T>
inline auto overlaps(T a0, T a1, T b0, T b1) -> bool
{
  normalize(a0, a1);
  normalize(b0, b1);
  return std::max(a0, b0) < std::min(a1, b1);
}

// Returns the geometric intersection. The result may be invalid; callers that
// have not already checked overlaps() should verify it with is_valid().
template <typename T>
inline auto intersection(T a0, T a1, T b0, T b1) -> Range<T>
{
  normalize(a0, a1);
  normalize(b0, b1);
  return {std::max(a0, b0), std::min(a1, b1)};
}

template <typename T>
inline auto midpoint(T coord0, T coord1) -> T
{
  return coord0 + (coord1 - coord0) / 2;
}

template <typename IntervalT>
inline auto subtract(
    const std::vector<IntervalT>& intervals,
    std::remove_cvref_t<decltype(std::declval<IntervalT>().a0)> cut_a0,
    std::remove_cvref_t<decltype(std::declval<IntervalT>().a1)> cut_a1) -> std::vector<IntervalT>
{
  using Coord = std::remove_cvref_t<decltype(std::declval<IntervalT>().a0)>;

  std::vector<IntervalT> next;
  normalize(cut_a0, cut_a1);

  for (const auto& interval : intervals) {
    if (!overlaps(interval.a0, interval.a1, cut_a0, cut_a1)) {
      next.push_back(interval);
      continue;
    }

    const IntervalT left{interval.a0, static_cast<Coord>(std::min(interval.a1, cut_a0))};
    if (is_valid(left.a0, left.a1)) {
      next.push_back(left);
    }

    const IntervalT right{static_cast<Coord>(std::max(interval.a0, cut_a1)), interval.a1};
    if (is_valid(right.a0, right.a1)) {
      next.push_back(right);
    }
  }

  return next;
}

template <typename IntervalT, typename Mergeable>
inline auto clip(
    const std::vector<IntervalT>& intervals,
    std::remove_cvref_t<decltype(std::declval<IntervalT>().a0)> clip_a0,
    std::remove_cvref_t<decltype(std::declval<IntervalT>().a1)> clip_a1,
    Mergeable mergeable) -> std::vector<IntervalT>
{
  std::vector<IntervalT> clipped;
  if (!is_valid(clip_a0, clip_a1)) {
    return clipped;
  }

  for (const auto& interval : intervals) {
    const auto a0 = std::max(clip_a0, interval.a0);
    const auto a1 = std::min(clip_a1, interval.a1);
    if (!is_valid(a0, a1)) {
      continue;
    }

    if (!clipped.empty() &&
        clipped.back().a1 == a0 &&
        mergeable(clipped.back(), interval)) {
      clipped.back().a1 = a1;
      continue;
    }

    IntervalT out = interval;
    out.a0 = a0;
    out.a1 = a1;
    clipped.push_back(std::move(out));
  }

  return clipped;
}

}  // namespace interval
}  // namespace ircx
