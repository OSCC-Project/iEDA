#pragma once

#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class LayerCoord : public PlanarCoord
{
 public:
  LayerCoord() = default;
  LayerCoord(const PlanarCoord& coord, const irt_int layer_idx = -1) : PlanarCoord(coord) { _layer_idx = layer_idx; }
  LayerCoord(const irt_int x, const irt_int y, const irt_int layer_idx = -1) : PlanarCoord(x, y) { _layer_idx = layer_idx; }
  ~LayerCoord() = default;
  bool operator==(const LayerCoord& other) const { return PlanarCoord::operator==(other) && _layer_idx == other._layer_idx; }
  bool operator!=(const LayerCoord& other) const { return !((*this) == other); }
  // getter
  PlanarCoord& get_planar_coord() { return (*this); }
  irt_int get_layer_idx() const { return _layer_idx; }
  // const getter
  const PlanarCoord& get_planar_coord() const { return (*this); }
  // setter
  void set_coord(const PlanarCoord& coord)
  {
    set_x(coord.get_x());
    set_y(coord.get_y());
  }
  using PlanarCoord::set_coord;
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  irt_int _layer_idx = -1;
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

}  // namespace irt
