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
 * @file Rect.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Axis-aligned rectangle value type for iCTS spatial geometry.
 */

#pragma once

#include <algorithm>
#include <optional>

#include "Point.hh"

namespace icts {

template <typename T>
class Rect
{
 public:
  using CoordType = T;

  Rect() = default;
  Rect(const Point<T>& lower_left, const Point<T>& upper_right)
      : _min_x(std::min(lower_left.get_x(), upper_right.get_x())),
        _min_y(std::min(lower_left.get_y(), upper_right.get_y())),
        _max_x(std::max(lower_left.get_x(), upper_right.get_x())),
        _max_y(std::max(lower_left.get_y(), upper_right.get_y()))
  {
  }
  Rect(T min_x, T min_y, T max_x, T max_y)
      : _min_x(std::min(min_x, max_x)), _min_y(std::min(min_y, max_y)), _max_x(std::max(min_x, max_x)), _max_y(std::max(min_y, max_y))
  {
  }

  auto get_min_x() const -> T { return _min_x; }
  auto get_min_y() const -> T { return _min_y; }
  auto get_max_x() const -> T { return _max_x; }
  auto get_max_y() const -> T { return _max_y; }

  auto get_lower_left() const -> Point<T> { return Point<T>(_min_x, _min_y); }
  auto get_upper_right() const -> Point<T> { return Point<T>(_max_x, _max_y); }

  auto contains(const Point<T>& point) const -> bool
  {
    return point.get_x() >= _min_x && point.get_x() <= _max_x && point.get_y() >= _min_y && point.get_y() <= _max_y;
  }

  auto overlaps(const Rect& other) const -> bool
  {
    return !(_max_x < other._min_x || other._max_x < _min_x || _max_y < other._min_y || other._max_y < _min_y);
  }

  auto intersect(const Rect& other) const -> std::optional<Rect>
  {
    if (!overlaps(other)) {
      return std::nullopt;
    }
    return Rect(std::max(_min_x, other._min_x), std::max(_min_y, other._min_y), std::min(_max_x, other._max_x),
                std::min(_max_y, other._max_y));
  }

  auto clamp(const Point<T>& point) const -> Point<T>
  {
    return Point<T>(std::clamp(point.get_x(), _min_x, _max_x), std::clamp(point.get_y(), _min_y, _max_y));
  }

  auto operator==(const Rect& rhs) const -> bool
  {
    return _min_x == rhs._min_x && _min_y == rhs._min_y && _max_x == rhs._max_x && _max_y == rhs._max_y;
  }

  auto operator!=(const Rect& rhs) const -> bool { return !(*this == rhs); }

 private:
  T _min_x{};
  T _min_y{};
  T _max_x{};
  T _max_y{};
};

}  // namespace icts
