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
 * @file Region.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Rectilinear region value type built from normalized rectangles.
 */

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

#include "Rect.hh"

namespace icts {

template <typename T>
class Region
{
 public:
  using RectType = Rect<T>;
  using PointType = Point<T>;

  Region() = default;
  explicit Region(const RectType& rect) { add_rect(rect); }
  explicit Region(std::vector<RectType> rects) : _rects(std::move(rects)) { normalize(); }

  auto empty() const -> bool { return _rects.empty(); }
  auto rects() const -> const std::vector<RectType>& { return _rects; }

  auto add_rect(const RectType& rect) -> void
  {
    _rects.push_back(rect);
    normalize();
  }

  auto contains(const PointType& point) const -> bool
  {
    return std::ranges::any_of(_rects, [&](const auto& rect) -> auto { return rect.contains(point); });
  }

  auto subtract(const RectType& blocked_rect) -> void
  {
    std::vector<RectType> remain_rects;
    remain_rects.reserve(_rects.size() * 4);
    for (const auto& rect : _rects) {
      auto pieces = subtractRect(rect, blocked_rect);
      std::ranges::move(pieces, std::back_inserter(remain_rects));
    }
    _rects = std::move(remain_rects);
    normalize();
  }

  auto subtract(const Region& blocked_region) -> void
  {
    for (const auto& rect : blocked_region.rects()) {
      subtract(rect);
    }
  }

  auto project_nearest(const PointType& point) const -> std::optional<PointType>
  {
    if (_rects.empty()) {
      return std::nullopt;
    }

    std::optional<PointType> best_point;
    auto best_distance = std::numeric_limits<T>::max();
    for (const auto& rect : _rects) {
      auto projected = rect.clamp(point);
      using std::abs;
      auto distance = abs(point.get_x() - projected.get_x()) + abs(point.get_y() - projected.get_y());
      if (!best_point.has_value() || distance < best_distance || (distance == best_distance && projected < best_point.value())) {
        best_point = projected;
        best_distance = distance;
      }
    }
    return best_point;
  }

 private:
  static auto subtractRect(const RectType& source, const RectType& blocked) -> std::vector<RectType>
  {
    auto overlap = source.intersect(blocked);
    if (!overlap.has_value()) {
      return {source};
    }

    const auto& cut = overlap.value();
    std::vector<RectType> pieces;
    pieces.reserve(4);

    if (source.get_min_y() < cut.get_min_y()) {
      pieces.emplace_back(source.get_min_x(), source.get_min_y(), source.get_max_x(), static_cast<T>(cut.get_min_y() - 1));
    }
    if (cut.get_max_y() < source.get_max_y()) {
      pieces.emplace_back(source.get_min_x(), static_cast<T>(cut.get_max_y() + 1), source.get_max_x(), source.get_max_y());
    }
    if (source.get_min_x() < cut.get_min_x()) {
      pieces.emplace_back(source.get_min_x(), cut.get_min_y(), static_cast<T>(cut.get_min_x() - 1), cut.get_max_y());
    }
    if (cut.get_max_x() < source.get_max_x()) {
      pieces.emplace_back(static_cast<T>(cut.get_max_x() + 1), cut.get_min_y(), source.get_max_x(), cut.get_max_y());
    }

    return pieces;
  }

  auto normalize() -> void
  {
    if (_rects.empty()) {
      return;
    }

    std::sort(_rects.begin(), _rects.end(), [](const RectType& lhs, const RectType& rhs) -> auto {
      if (lhs.get_min_x() != rhs.get_min_x()) {
        return lhs.get_min_x() < rhs.get_min_x();
      }
      if (lhs.get_min_y() != rhs.get_min_y()) {
        return lhs.get_min_y() < rhs.get_min_y();
      }
      if (lhs.get_max_x() != rhs.get_max_x()) {
        return lhs.get_max_x() < rhs.get_max_x();
      }
      return lhs.get_max_y() < rhs.get_max_y();
    });

    bool merged = true;
    while (merged) {
      merged = false;
      std::vector<RectType> normalized;
      for (const auto& rect : _rects) {
        bool merged_into_existing = false;
        for (auto& existing : normalized) {
          if (canMerge(existing, rect)) {
            existing = merge(existing, rect);
            merged = true;
            merged_into_existing = true;
            break;
          }
        }
        if (!merged_into_existing) {
          normalized.push_back(rect);
        }
      }
      _rects = std::move(normalized);
    }
  }

  static auto canMerge(const RectType& lhs, const RectType& rhs) -> bool
  {
    const bool share_x_span = lhs.get_min_x() == rhs.get_min_x() && lhs.get_max_x() == rhs.get_max_x();
    const bool share_y_span = lhs.get_min_y() == rhs.get_min_y() && lhs.get_max_y() == rhs.get_max_y();
    const bool y_overlap_or_touch
        = lhs.get_min_y() <= static_cast<T>(rhs.get_max_y() + 1) && rhs.get_min_y() <= static_cast<T>(lhs.get_max_y() + 1);
    const bool x_overlap_or_touch
        = lhs.get_min_x() <= static_cast<T>(rhs.get_max_x() + 1) && rhs.get_min_x() <= static_cast<T>(lhs.get_max_x() + 1);
    return (share_x_span && y_overlap_or_touch) || (share_y_span && x_overlap_or_touch);
  }

  static auto merge(const RectType& lhs, const RectType& rhs) -> RectType
  {
    return RectType(std::min(lhs.get_min_x(), rhs.get_min_x()), std::min(lhs.get_min_y(), rhs.get_min_y()),
                    std::max(lhs.get_max_x(), rhs.get_max_x()), std::max(lhs.get_max_y(), rhs.get_max_y()));
  }

  std::vector<RectType> _rects;
};

}  // namespace icts
