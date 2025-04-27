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

#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "PlanarRect.hpp"

namespace idrc {

class LayerRect : public PlanarRect
{
 public:
  LayerRect() = default;
  LayerRect(const PlanarRect& rect_2d, const int32_t layer_idx = -1) : PlanarRect(rect_2d) { _layer_idx = layer_idx; }
  LayerRect(const PlanarCoord& ll, const PlanarCoord& ur, const int32_t layer_idx = -1) : PlanarRect(ll, ur) { _layer_idx = layer_idx; }
  LayerRect(const int32_t ll_x, const int32_t ll_y, const int32_t ur_x, const int32_t ur_y, const int32_t layer_idx = -1) : PlanarRect(ll_x, ll_y, ur_x, ur_y)
  {
    _layer_idx = layer_idx;
  }
  ~LayerRect() = default;

  bool operator==(const LayerRect& other) const { return PlanarRect::operator==(other) && _layer_idx == other._layer_idx; }
  bool operator!=(const LayerRect& other) const { return !((*this) == other); }
  // getter
  PlanarRect& get_rect() { return (*this); }
  int32_t get_layer_idx() const { return _layer_idx; }
  // const getter
  const PlanarRect& get_rect() const { return (*this); }
  // setter
  void set_rect(const PlanarRect& rect)
  {
    set_ll(rect.get_ll());
    set_ur(rect.get_ur());
  }
  void set_rect(const PlanarCoord& ll, const PlanarCoord& ur)
  {
    set_ll(ll);
    set_ur(ur);
  }
  void set_rect(const int32_t ll_x, const int32_t ll_y, const int32_t ur_x, const int32_t ur_y)
  {
    set_ll(ll_x, ll_y);
    set_ur(ur_x, ur_y);
  }
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  int32_t _layer_idx = -1;
};

struct CmpLayerRectByXASC
{
  bool operator()(const LayerRect& a, const LayerRect& b) const
  {
    if (a.get_ll() == b.get_ll()) {
      return CmpLayerCoordByXASC()(LayerCoord(a.get_ur(), a.get_layer_idx()), LayerCoord(b.get_ur(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByXASC()(LayerCoord(a.get_ll(), a.get_layer_idx()), LayerCoord(b.get_ll(), b.get_layer_idx()));
    }
  }
};

struct CmpLayerRectByYASC
{
  bool operator()(const LayerRect& a, const LayerRect& b) const
  {
    if (a.get_ll() == b.get_ll()) {
      return CmpLayerCoordByYASC()(LayerCoord(a.get_ur(), a.get_layer_idx()), LayerCoord(b.get_ur(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByYASC()(LayerCoord(a.get_ll(), a.get_layer_idx()), LayerCoord(b.get_ll(), b.get_layer_idx()));
    }
  }
};
struct CmpLayerRectByLayerASC
{
  bool operator()(const LayerRect& a, const LayerRect& b) const
  {
    if (a.get_ll() == b.get_ll()) {
      return CmpLayerCoordByLayerASC()(LayerCoord(a.get_ur(), a.get_layer_idx()), LayerCoord(b.get_ur(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByLayerASC()(LayerCoord(a.get_ll(), a.get_layer_idx()), LayerCoord(b.get_ll(), b.get_layer_idx()));
    }
  }
};

}  // namespace idrc
