#include "DrcNet.h"

namespace idrc {

std::set<int> DrcNet::get_layer_id_list()
{
  std::set<int> layer_id_list;
  for (auto& [layerId, polyset] : _layer_to_routing_polygon_set) {
    layer_id_list.insert(layerId);
  }
  for (auto& [layerId, polyset] : _layer_to_pin_polygon_set) {
    layer_id_list.insert(layerId);
  }
  return layer_id_list;
}

DrcPolygon* DrcNet::add_merge_polygon(int routingLayer, const PolygonWithHoles& poly)
{
  std::unique_ptr<DrcPolygon> drcPolygon = std::make_unique<DrcPolygon>(routingLayer, poly);
  DrcPolygon* drc_polygon = drcPolygon.get();
  _layer_to_merge_polygon_list[routingLayer].push_back(std::move(drcPolygon));
  // assert(drc_polygon != nullptr);
  return drc_polygon;
}

void DrcNet::clear_layer_to_routing_rects_map()
{
  for (auto& [layer_id, routing_rect_list] : _layer_to_routing_rects_map) {
    for (auto routing_rect : routing_rect_list) {
      if (routing_rect != nullptr) {
        delete routing_rect;
        routing_rect = nullptr;
      }
    }
  }
  _layer_to_routing_rects_map.clear();
}
void DrcNet::clear_layer_to_pin_rects_map()
{
  for (auto& [layer_id, pin_rect_list] : _layer_to_pin_rects_map) {
    for (auto pin_rect : pin_rect_list) {
      if (pin_rect != nullptr) {
        delete pin_rect;
        pin_rect = nullptr;
      }
    }
  }
  _layer_to_pin_rects_map.clear();
}
void DrcNet::clear_layer_to_cut_rects_map()
{
  for (auto& [layer_id, cut_rect_list] : _layer_to_cut_rects_map) {
    for (auto cut_rect : cut_rect_list) {
      if (cut_rect != nullptr) {
        delete cut_rect;
        cut_rect = nullptr;
      }
    }
  }
  _layer_to_cut_rects_map.clear();
}

void DrcNet::addPoly(PolygonWithHoles shape, int layer_id)
{
  DrcPoly* poly = new DrcPoly();
  poly->setNet(this);
  poly->set_layer_id(layer_id);
  DrcPolygon* polygon = new DrcPolygon(shape, layer_id, poly, this);
  poly->setPolygon(polygon);
  // polygon->set_boost_polygon(shape);
  // std::cout << (*poly->getPolygon()->begin()).x() << "       " << (*poly->getPolygon()->begin()).y()
  //           << std::endl;
  // for (auto outerIt = polygon->begin(); outerIt != polygon->end(); outerIt++) {
  //   auto bp1 = *outerIt;
  //   std::cout << "BP::::::::" << bp1.x() << "          " << bp1.y() << std::endl;
  // }
  // auto poly = std::make_unique<DrcPoly>(shape, layer_id, this);
  // poly->set_layer_id(_layer_to_route_polys_map[layer_id].size());
  _route_polys_list[layer_id].push_back(std::unique_ptr<DrcPoly>(poly));
}

void DrcNet::addPoly(DrcPoly* poly)
{
  int layer_id = poly->get_layer_id();
  _route_polys_list[layer_id].push_back(std::unique_ptr<DrcPoly>(poly));
}

}  // namespace idrc
