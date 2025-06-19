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

#include "DRCHeader.hpp"
#include "LayerCoord.hpp"

namespace idrc {

template <typename T>
class Segment
{
 public:
  Segment() = default;
  Segment(const T& first, const T& second)
  {
    _first = first;
    _second = second;
  }
  ~Segment() = default;
  // getter
  T& get_first() { return _first; }
  T& get_second() { return _second; }
  const T& get_first() const { return _first; }
  const T& get_second() const { return _second; }
  // setter
  void set_first(const T& x) { _first = x; }
  void set_second(const T& y) { _second = y; }
  // function

 private:
  T _first;
  T _second;
};

struct SortSegmentInnerXASC
{
  void operator()(Segment<PlanarCoord>& a) const
  {
    PlanarCoord& first_coord = a.get_first();
    PlanarCoord& second_coord = a.get_second();
    if (CmpPlanarCoordByXASC()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct SortSegmentInnerYASC
{
  void operator()(Segment<PlanarCoord>& a) const
  {
    PlanarCoord& first_coord = a.get_first();
    PlanarCoord& second_coord = a.get_second();
    if (CmpPlanarCoordByYASC()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct CmpSegmentXASC
{
  bool operator()(const Segment<PlanarCoord>& a, const Segment<PlanarCoord>& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpPlanarCoordByXASC()(a.get_first(), b.get_first());
    } else {
      return CmpPlanarCoordByXASC()(a.get_second(), b.get_second());
    }
  }
};

struct CmpSegmentYASC
{
  bool operator()(const Segment<PlanarCoord>& a, const Segment<PlanarCoord>& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpPlanarCoordByYASC()(a.get_first(), b.get_first());
    } else {
      return CmpPlanarCoordByYASC()(a.get_second(), b.get_second());
    }
  }
};

}  // namespace idrc
