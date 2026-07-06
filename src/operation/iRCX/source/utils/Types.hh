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

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/polygon/gtl.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

namespace ircx {

// -------------------------
// Fundamental scalar types
// -------------------------
using U8  = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

using I8  = std::int8_t;
using I16 = std::int16_t;
using I32 = std::int32_t;
using I64 = std::int64_t;

using F32 = float;
using F64 = double;

using Str = std::string;
using Size = std::size_t;

using Dbu    = I32;   // database unit
using Micron = F64;   // micron unit

namespace unit {

inline auto to_micron(Dbu value, Dbu dbu_per_micron) -> Micron
{
  return static_cast<Micron>(value) / static_cast<Micron>(dbu_per_micron);
}

inline auto to_dbu(Micron value, Dbu dbu_per_micron) -> Dbu
{
  return static_cast<Dbu>(std::llround(value * static_cast<Micron>(dbu_per_micron)));
}

}  // namespace unit

template <typename T>
inline constexpr int BitNum = std::numeric_limits<T>::digits;

template <typename T>
inline constexpr T MaxVal = std::numeric_limits<T>::max();

template <typename T>
inline constexpr T MinVal = std::numeric_limits<T>::min();

template <typename T>
inline constexpr T InfinityVal = std::numeric_limits<T>::infinity();

// Common "invalid" sentinels
inline constexpr Dbu  kMaxDbu  = MaxVal<Dbu>;
inline constexpr Size kMaxSize = MaxVal<Size>;
inline constexpr Micron kMaxMicron = MaxVal<Micron>;

// Sentinel net_id for special-net (power/ground) edges.
// Distinct from all valid net ids and from kMaxSize.
inline constexpr Size kSpecialNetId = kMaxSize - 1;

// -------------------------
// LineSegment geometry
// -------------------------

// A wire segment projected onto one axis.
//   is_horz : true -> horizontal wire (coord is Y, range is X)
//   coord   : the constant coordinate (Y for horz, X for vert)
//   lo, hi  : range along the wire direction
template <typename T = Dbu>
struct LineSegment {
  bool is_horz{false};
  T coord{};
  T lo{};
  T hi{};
};

using LineSegmentI = LineSegment<Dbu>;

// -------------------------
// Boost namespace aliases
// -------------------------
namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;
namespace gtl = boost::polygon;

// -------------------------
// Geometry type aliases
// -------------------------
using BgPointI   = bg::model::point<Dbu, 2, bg::cs::cartesian>;
using BgBoxI     = bg::model::box<BgPointI>;
using BgPolygonI = bg::model::polygon<BgPointI>;

using GtlPointI   = gtl::point_data<Dbu>;
using GtlRectI    = gtl::rectangle_data<Dbu>;
using GtlPolyI    = gtl::polygon_90_data<Dbu>;
using GtlPolySetI = gtl::polygon_90_set_data<Dbu>;

using GtlPointF   = gtl::point_data<F64>;
using GtlRectF    = gtl::rectangle_data<F64>;
using GtlPolyF    = gtl::polygon_90_data<F64>;
using GtlPolySetF = gtl::polygon_90_set_data<F64>;
}  // namespace ircx
