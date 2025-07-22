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
#include "PlanarCoord.hpp"

namespace idrc {

class LayerCoord : public PlanarCoord
{
 public:
  LayerCoord() = default;
  LayerCoord(const PlanarCoord& coord, const int32_t layer_idx = -1) : PlanarCoord(coord) { _layer_idx = layer_idx; }
  LayerCoord(const int32_t x, const int32_t y, const int32_t layer_idx = -1) : PlanarCoord(x, y) { _layer_idx = layer_idx; }
  ~LayerCoord() = default;
  bool operator==(const LayerCoord& other) const { return PlanarCoord::operator==(other) && _layer_idx == other._layer_idx; }
  bool operator!=(const LayerCoord& other) const { return !((*this) == other); }
  // getter
  PlanarCoord& get_planar_coord() { return (*this); }
  int32_t get_layer_idx() const { return _layer_idx; }
  // const getter
  const PlanarCoord& get_planar_coord() const { return (*this); }
  // setter
  void set_coord(const PlanarCoord& coord)
  {
    set_x(coord.get_x());
    set_y(coord.get_y());
  }
  using PlanarCoord::set_coord;
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  int32_t _layer_idx = -1;
};

struct CmpLayerCoordByXASC
{
  bool operator()(const LayerCoord& a, const LayerCoord& b) const
  {
    if (a.get_x() != b.get_x()) {
      return a.get_x() < b.get_x();
    } else {
      return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_layer_idx() < b.get_layer_idx();
    }
  }
};

struct CmpLayerCoordByYASC
{
  bool operator()(const LayerCoord& a, const LayerCoord& b) const
  {
    if (a.get_y() != b.get_y()) {
      return a.get_y() < b.get_y();
    } else {
      return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_layer_idx() < b.get_layer_idx();
    }
  }
};

struct CmpLayerCoordByLayerASC
{
  bool operator()(const LayerCoord& a, const LayerCoord& b) const
  {
    if (a.get_layer_idx() != b.get_layer_idx()) {
      return a.get_layer_idx() < b.get_layer_idx();
    } else {
      return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_y() < b.get_y();
    }
  }
};

}  // namespace idrc
