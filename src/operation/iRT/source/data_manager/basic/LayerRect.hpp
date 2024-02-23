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

#include "LayerRect.hpp"
#include "PlanarRect.hpp"

namespace irt {

class LayerRect : public PlanarRect
{
 public:
  LayerRect() = default;
  LayerRect(const PlanarRect& rect_2d, const int32_t layer_idx = -1) : PlanarRect(rect_2d) { _layer_idx = layer_idx; }
  LayerRect(const PlanarCoord& lb, const PlanarCoord& rt, const int32_t layer_idx = -1) : PlanarRect(lb, rt) { _layer_idx = layer_idx; }
  LayerRect(const int32_t lb_x, const int32_t lb_y, const int32_t rt_x, const int32_t rt_y, const int32_t layer_idx = -1)
      : PlanarRect(lb_x, lb_y, rt_x, rt_y)
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
    set_lb(rect.get_lb());
    set_rt(rect.get_rt());
  }
  void set_rect(const PlanarCoord& lb, const PlanarCoord& rt)
  {
    set_lb(lb);
    set_rt(rt);
  }
  void set_rect(const int32_t lb_x, const int32_t lb_y, const int32_t rt_x, const int32_t rt_y)
  {
    set_lb(lb_x, lb_y);
    set_rt(rt_x, rt_y);
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
    if (a.get_lb() == b.get_lb()) {
      return CmpLayerCoordByXASC()(LayerCoord(a.get_rt(), a.get_layer_idx()), LayerCoord(b.get_rt(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByXASC()(LayerCoord(a.get_lb(), a.get_layer_idx()), LayerCoord(b.get_lb(), b.get_layer_idx()));
    }
  }
};

struct CmpLayerRectByYASC
{
  bool operator()(const LayerRect& a, const LayerRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpLayerCoordByYASC()(LayerCoord(a.get_rt(), a.get_layer_idx()), LayerCoord(b.get_rt(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByYASC()(LayerCoord(a.get_lb(), a.get_layer_idx()), LayerCoord(b.get_lb(), b.get_layer_idx()));
    }
  }
};
struct CmpLayerRectByLayerASC
{
  bool operator()(const LayerRect& a, const LayerRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpLayerCoordByLayerASC()(LayerCoord(a.get_rt(), a.get_layer_idx()), LayerCoord(b.get_rt(), b.get_layer_idx()));
    } else {
      return CmpLayerCoordByLayerASC()(LayerCoord(a.get_lb(), a.get_layer_idx()), LayerCoord(b.get_lb(), b.get_layer_idx()));
    }
  }
};

}  // namespace irt
