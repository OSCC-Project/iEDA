#ifndef IDRC_SRC_DB_DRC_POLYGON_H_
#define IDRC_SRC_DB_DRC_POLYGON_H_

#include <memory>

#include "BoostType.h"
#include "DrcEdge.h"
#include "DrcNet.h"
#include "DrcPoly.hpp"
namespace idrc {
class DrcPoly;
class DrcNet;
class DrcPolygon : public PolygonWithHoles
{
 public:
  DrcPolygon() : _net_id(-1), _layer_id(-1), _poly(nullptr), _net(nullptr) {}
  DrcPolygon(int layerId, DrcPoly* polyIn, DrcNet* netIn) : _net_id(-1), _layer_id(layerId), _poly(polyIn), _net(netIn) {}
  DrcPolygon(int layerId, DrcPoly* polyIn, int netId) : _net_id(netId), _layer_id(layerId), _poly(polyIn), _net(nullptr) {}
  explicit DrcPolygon(const PolygonWithHoles& polygon) : _net_id(-1), _layer_id(-1), _poly(nullptr), _polygon(polygon) {}
  DrcPolygon(int layerId, const PolygonWithHoles& polygon) : _net_id(), _layer_id(layerId), _net(nullptr), _polygon(polygon) {}
  DrcPolygon(int layerId, const PolygonWithHoles& polygon, DrcNet* net)
      : _net_id(), _layer_id(layerId), _poly(), _net(nullptr), _polygon(polygon)
  {
  }
  DrcPolygon(const PolygonWithHoles& shapeIn, int layerIn, DrcPoly* polyIn, DrcNet* netIn)
      : PolygonWithHoles(shapeIn), _net_id(-1), _layer_id(layerIn), _poly(polyIn), _net(netIn), _polygon(shapeIn)
  {
  }
  DrcPolygon(const PolygonWithHoles& shapeIn, int layerIn, DrcPoly* polyIn, int net_id)
      : PolygonWithHoles(shapeIn), _net_id(net_id), _layer_id(layerIn), _poly(polyIn), _net(nullptr), _polygon(shapeIn)
  {
  }
  ~DrcPolygon() {}

  // setter
  void add_polygon_edge(std::unique_ptr<DrcEdge>& edge) { _edge_list.push_back(std::move(edge)); }
  void set_layer_id(int layerId) { _layer_id = layerId; }
  void set_net_id(int netId) { _net_id = netId; }
  void set_boost_polygon(PolygonWithHoles in) { _polygon = in; }
  // getter
  int get_net_id() const { return _net_id; }
  int get_layer_id() const { return _layer_id; }
  PolygonWithHoles get_polygon() { return _polygon; }
  // function

 private:
  int _net_id;
  int _layer_id;
  DrcPoly* _poly;
  DrcNet* _net;
  PolygonWithHoles _polygon;
  std::vector<std::unique_ptr<DrcEdge>> _edge_list;
  std::vector<std::unique_ptr<DrcRect>> _rect_list;
};
}  // namespace idrc

#endif