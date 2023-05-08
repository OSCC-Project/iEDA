#ifndef IDRC_SRC_DB_DRCNET_H_
#define IDRC_SRC_DB_DRCNET_H_

#include <assert.h>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "BoostType.h"
#include "DrcEdge.h"
#include "DrcPoly.hpp"
#include "DrcPolygon.h"
#include "DrcRect.h"

namespace idrc {

class DrcPoly;

class DrcNet
{
 public:
  DrcNet() {}
  explicit DrcNet(int netId) { _net_id = netId; }
  DrcNet(DrcNet&& other) = default;
  ~DrcNet()
  {
    clear_layer_to_routing_rects_map();
    clear_layer_to_pin_rects_map();
    clear_layer_to_cut_rects_map();
  }
  DrcNet& operator=(DrcNet&& other) = default;
  // setter
  void set_net_id(int net_id) { _net_id = net_id; }
  void add_routing_rect(int routingLayerId, DrcRect* rect) { _layer_to_routing_rects_map[routingLayerId].push_back(rect); }
  void add_pin_rect(int routingLayerId, DrcRect* rect) { _layer_to_pin_rects_map[routingLayerId].push_back(rect); }
  void add_rect_edge(int routingLayerId, DrcEdge* edge) { _layer_to_rect_edges_map[routingLayerId].push_back(edge); }
  void add_cut_rect(int cutLayerId, DrcRect* rect) { _layer_to_cut_rects_map[cutLayerId].push_back(rect); }
  // getter
  int get_net_id() const { return _net_id; }
  std::vector<DrcRect*>& get_layer_routing_rects(int routingLayerId) { return _layer_to_routing_rects_map[routingLayerId]; }
  std::vector<DrcRect*>& get_layer_pin_rects(int routingLayerId) { return _layer_to_pin_rects_map[routingLayerId]; }
  std::vector<DrcEdge*>& get_layer_rect_edges(int routingLayerId) { return _layer_to_rect_edges_map[routingLayerId]; }
  std::vector<DrcRect*>& get_layer_cut_rects(int cutLayerId) { return _layer_to_cut_rects_map[cutLayerId]; }
  std::map<int, PolygonSet>& get_routing_polygon_set_map() { return _layer_to_routing_polygon_set; }
  PolygonSet& get_routing_polygon_set_by_id(int routing_layer_id) { return _layer_to_routing_polygon_set[routing_layer_id]; }

  std::map<int, std::vector<DrcRect*>>& get_layer_to_routing_rects_map() { return _layer_to_routing_rects_map; }
  std::map<int, std::vector<DrcRect*>>& get_layer_to_pin_rects_map() { return _layer_to_pin_rects_map; }
  std::map<int, std::vector<DrcRect*>>& get_layer_to_cut_rects_map() { return _layer_to_cut_rects_map; }
  std::map<int, std::set<DrcPoly*>>& get_layer_to_routing_poly_map() { return _layer_to_poly_set_map; }
  std::set<DrcPoly*>& get_routing_poly_set(int layer_id) { return _layer_to_poly_set_map[layer_id]; }
  // function
  // clear
  void clear_layer_to_routing_rects_map();

  void clear_layer_to_pin_rects_map();

  void clear_layer_to_cut_rects_map();

  void clear_layer_to_routing_polygon_set() { _layer_to_routing_polygon_set.clear(); }
  void clear_layer_to_pin_polygon_set() { _layer_to_pin_polygon_set.clear(); }
  void clear_layer_to_merge_polygon_list() { _layer_to_merge_polygon_list.clear(); }

  // function
  void add_pin_rect(int routingLayer, const BoostRect& rect) { _layer_to_pin_polygon_set[routingLayer] += rect; }
  void add_routing_rect(int routingLayer, const BoostRect& rect)
  {
    // std::cout << this->get_net_id() << std::endl;
    _layer_to_routing_polygon_set[routingLayer] += rect;
  }
  void add_merge_polygon(int routingLayer, std::unique_ptr<DrcPolygon>& poly)
  {
    _layer_to_merge_polygon_list[routingLayer].push_back(std::move(poly));
  }

  void addPoly(PolygonWithHoles shape, int layer_id);
  void addPoly(DrcPoly* poly);
  DrcPolygon* add_merge_polygon(int routingLayer, const PolygonWithHoles& poly);

  // getter
  std::vector<std::unique_ptr<DrcPoly>>& get_route_polys(int layer_id) { return _route_polys_list[layer_id]; }
  std::map<int, std::vector<std::unique_ptr<DrcPoly>>>& get_route_polys_list() { return _route_polys_list; }
  std::map<int, bp::polygon_90_set_data<int>>& get_layer_to_routing_polygon_set() { return _layer_to_routing_polygon_set; }
  // BoostPolygon& get_routing_polygon_set_by_id(int layer_id){return _layer_to_routing_polygon_set[layer_id];}
  std::map<int, bp::polygon_90_set_data<int>>& get_layer_to_pin_polygon_set() { return _layer_to_pin_polygon_set; }
  bp::polygon_90_set_data<int>& get_routing_polygon_set(int layerId) { return _layer_to_routing_polygon_set[layerId]; }
  bp::polygon_90_set_data<int>& get_pin_polygon_set(int layerId) { return _layer_to_pin_polygon_set[layerId]; }
  std::set<int> get_layer_id_list();

 private:
  int _net_id;
  // segemnt and viaMetal rect
  std::map<int, std::vector<DrcRect*>> _layer_to_routing_rects_map;
  // pin rect is fixed
  std::map<int, std::vector<DrcRect*>> _layer_to_pin_rects_map;

  std::map<int, std::vector<std::unique_ptr<DrcPoly>>> _route_polys_list;

  ////////////////////////////
  //工程上未用到，multi-patterning相关
  std::map<int, bp::polygon_90_set_data<int>> _layer_to_routing_polygon_set;
  std::map<int, bp::polygon_90_set_data<int>> _layer_to_pin_polygon_set;

  std::map<int, std::vector<std::unique_ptr<DrcPolygon>>> _layer_to_merge_polygon_list;  // merge

  // api
  std::map<int, std::set<DrcPoly*>> _layer_to_poly_set_map;
  ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////
  // not use now
  // cut rect only for cut layer
  std::map<int, std::vector<DrcRect*>> _layer_to_cut_rects_map;
  // rect edge from routing rect and pin rect
  std::map<int, std::vector<DrcEdge*>> _layer_to_rect_edges_map;
};
}  // namespace idrc

#endif