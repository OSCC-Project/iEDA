#pragma once

#include <vector>

#include "DrcCorner.hpp"
#include "DrcNet.h"
#include "DrcPolygon.h"

namespace idrc {
class DrcNet;
class DrcPolygon;
class DrcPoly
{
 public:
  DrcPoly() : _layer_id(-1), _net_id(-1), _polygon(nullptr), _net(nullptr) /*, dirty(false)*/, _edges(), _corners(), _max_rects() {}
  DrcPoly(const PolygonWithHoles& shape_in, int layer_num_in, DrcNet* net_in)
      : _layer_id(-1),
        _net_id(-1),
        _polygon(std::make_unique<DrcPolygon>(shape_in, layer_num_in, this, net_in).get()),
        _net(net_in) /*, dirty(true)*/,
        _edges(),
        _corners(),
        _max_rects()
  {
  }
  DrcPoly(const PolygonWithHoles& shape_in, int layer_num_in, int net_id)
      : _layer_id(-1),
        _net_id(net_id),
        _polygon(std::make_unique<DrcPolygon>(shape_in, layer_num_in, this, net_id).get()),
        _net(nullptr),
        _edges(),
        _corners(),
        _max_rects()

  {
  }

  int get_layer_id() const { return _layer_id; }
  DrcPolygon* getPolygon() const { return _polygon; }
  DrcNet* getNet() const { return _net; }
  int getNetId() const { return _net_id; }
  std::set<DrcRect*>& getScopes() { return _scope_set; }

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& getEdges() { return _edges; }

  void setNet(DrcNet* in) { _net = in; }
  void setNetId(int net_id) { _net_id = net_id; }
  void setPolygon(DrcPolygon* in) { _polygon = in; }

  void set_layer_id(int in) { _layer_id = in; }

  void addScope(DrcRect* scope_rect) { _scope_set.insert(scope_rect); }
  void addEdges(std::vector<std::unique_ptr<DrcEdge>>& in) { _edges.push_back(std::move(in)); }
  void addCorners(std::vector<std::unique_ptr<DrcCorner>>& in) { _corners.push_back(std::move(in)); }

 private:
  int _layer_id;
  int _net_id;
  DrcPolygon* _polygon;
  DrcNet* _net;
  std::vector<std::vector<std::unique_ptr<DrcEdge>>> _edges;
  std::vector<std::vector<std::unique_ptr<DrcCorner>>> _corners;

  std::set<DrcRect*> _scope_set;

  std::vector<DrcRect*> _max_rects;
};
}  // namespace idrc
