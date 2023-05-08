#pragma once

#include "LayerRect.hpp"
#include "PlanarRect.hpp"

namespace irt {

class LayerRect : public PlanarRect
{
 public:
  LayerRect() = default;
  LayerRect(const PlanarRect& rect_2d, const irt_int layer_idx = -1) : PlanarRect(rect_2d) { _layer_idx = layer_idx; }
  LayerRect(const PlanarCoord& lb, const PlanarCoord& rt, const irt_int layer_idx = -1) : PlanarRect(lb, rt) { _layer_idx = layer_idx; }
  LayerRect(const irt_int lb_x, const irt_int lb_y, const irt_int rt_x, const irt_int rt_y, const irt_int layer_idx = -1)
      : PlanarRect(lb_x, lb_y, rt_x, rt_y)
  {
    _layer_idx = layer_idx;
  }
  ~LayerRect() = default;

  bool operator==(const LayerRect& other) const { return PlanarRect::operator==(other) && _layer_idx == other._layer_idx; }
  bool operator!=(const LayerRect& other) const { return !((*this) == other); }
  // getter
  PlanarRect& get_rect() { return (*this); }
  irt_int get_layer_idx() const { return _layer_idx; }
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
  void set_rect(const irt_int lb_x, const irt_int lb_y, const irt_int rt_x, const irt_int rt_y)
  {
    set_lb(lb_x, lb_y);
    set_rt(rt_x, rt_y);
  }
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  irt_int _layer_idx = -1;
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
