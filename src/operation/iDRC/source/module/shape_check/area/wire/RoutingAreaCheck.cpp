#include "RoutingAreaCheck.h"

namespace idrc {

bool RoutingAreaCheck::check(DrcNet* target_net)
{
  checkArea(target_net);
  return _check_result;
}

bool RoutingAreaCheck::check(DrcPoly* target_poly)
{
  checkArea(target_poly);
  return _check_result;
}

void RoutingAreaCheck::checkArea(DrcNet* target_net)
{
  for (auto& [layer_id, target_polys] : target_net->get_route_polys_list()) {
    for (auto& target_poly : target_polys) {
      checkArea(target_poly.get());
    }
  }
}

void RoutingAreaCheck::checkArea(DrcPoly* target_poly)
{
  int layer_id = target_poly->get_layer_id();
  _lef58_rules = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_area_rule_list();

  // if layer dont have area rule,skip check
  if ((_lef58_rules.size() == 0) && (_tech->get_drc_routing_layer_list()[layer_id]->get_min_area() <= 0)) {
    return;
  }
  if (!checkMinimumArea(target_poly)) {
    if (_interact_with_op) {
      _region_query->addViolation(ViolationType::kRoutingArea);
      addSpot(target_poly);
      _check_result = false;
    } else {
      // add_spot();
    }
    return;
  }
  int size = _lef58_rules.size();
  for (_rule_index = 0; _rule_index < size; _rule_index++) {
    if (!checkLef58Area(target_poly)) {
      if (_interact_with_op) {
        _region_query->addViolation(ViolationType::kRoutingArea);
        addSpot(target_poly);
        _check_result = false;
      } else {
        // add_spot();
      }
      return;
    }
  }
}

void RoutingAreaCheck::addSpot(DrcPoly* target_poly)
{
  auto box = DRCUtil::getPolyBoundingBox(target_poly);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = target_poly->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  spot->set_net_id(target_poly->getNetId());
  spot->set_vio_type(ViolationType::kRoutingArea);
  spot->setCoordinate(box.min_corner().x(), box.min_corner().y(), box.max_corner().x(), box.max_corner().y());
  _region_query->_min_area_spot_list.emplace_back(spot);
}

bool RoutingAreaCheck::checkLef58Area(DrcPoly* target_poly)
{
  int require_area = _lef58_rules[_rule_index]->get_min_area();
  auto boost_polygon = target_poly->getPolygon()->get_polygon();

  if (bp::area(boost_polygon) >= require_area) {
    return true;
  }
  if (_lef58_rules[_rule_index]->get_except_edge_length()) {
    if (_lef58_rules[_rule_index]->get_except_edge_length()->get_min_edge_length().has_value()) {
      if (checkMinEdge(target_poly)) {
        return true;
      }
    }
    if (checkMaxEdge(target_poly)) {
      return true;
    }
  }

  if (_lef58_rules[_rule_index]->get_except_min_size().size() != 0) {
    if (checkMinSize(target_poly)) {
      return true;
    }
  }
  return false;
}

bool RoutingAreaCheck::checkMinEdge(DrcPoly* target_poly)
{
  int min_length = _lef58_rules[_rule_index]->get_except_edge_length()->get_min_edge_length().value();
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if ((edge->getLength() > min_length)) {
        return false;
      }
    }
  }
  return true;
}

bool RoutingAreaCheck::checkMaxEdge(DrcPoly* target_poly)
{
  int max_length = _lef58_rules[_rule_index]->get_except_edge_length()->get_max_edge_length();
  for (auto& edges : target_poly->getEdges()) {
    for (auto& edge : edges) {
      if ((edge->getLength() >= max_length)) {
        return true;
      }
    }
  }
  return false;
}

bool RoutingAreaCheck::checkMinSize(DrcPoly* target_poly)
{
  std::vector<bp::rectangle_data<int>> rects;
  auto boost_polygon = target_poly->getPolygon()->get_polygon();
  bp::get_max_rectangles(rects, boost_polygon);
  for (auto& rect : rects) {
    int x_edge_length = bp::xh(rect) - bp::xl(rect);
    int y_edge_length = bp::yh(rect) - bp::yl(rect);
    int width = std::min(x_edge_length, y_edge_length);
    int length = std::max(x_edge_length, y_edge_length);
    for (auto& min_size : _lef58_rules[_rule_index]->get_except_min_size()) {
      int require_width = min_size.get_min_width();
      int require_length = min_size.get_min_length();
      if (width >= require_width && length >= require_length) {
        return true;
      }
    }
  }
  return false;
}

bool RoutingAreaCheck::checkMinimumArea(DrcPoly* target_poly)
{
  int layer_id = target_poly->get_layer_id();
  int require_area = _tech->get_drc_routing_layer_list()[layer_id]->get_min_area();
  auto boost_polygon = target_poly->getPolygon()->get_polygon();
  if (bp::area(boost_polygon) >= require_area) {
    return true;
  }
  return false;
}

void RoutingAreaCheck::add_spot()
{
  // TODO
}
// /**
//  * @brief 将目标检查线网中的各个矩形包括Via，segment，Pin通过Boost多边形接口合成多边形
//  *
//  * @param target_net
//  */
// void RoutingAreaCheck::initLayerPolygonSet(DrcNet* target_net)
// {
//   initLayerToPolygonSetFromRoutingRects(target_net);
//   initLayerToPolygonSetFromPinRects(target_net);
// }

// /**
//  * @brief 将目标检查线网中绕线矩形包括Via，segment合成多边形
//  *
//  * @param target_net
//  */
// void RoutingAreaCheck::initLayerToPolygonSetFromRoutingRects(DrcNet* target_net)
// {
//   for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_routing_rects_map()) {
//     for (auto routing_rect : routing_rect_list) {
//       BoostRect routingRect = DRCUtil::getBoostRect(routing_rect);
//       _layer_to_polygons_map[layerId] += routingRect;
//     }
//   }
// }

// /**
//  * @brief 将目标检查线网中Pin矩形参与到多边形的合成过程中
//  *
//  * @param target_net
//  */
// void RoutingAreaCheck::initLayerToPolygonSetFromPinRects(DrcNet* target_net)
// {
//   for (auto& [layerId, pin_rect_list] : target_net->get_layer_to_pin_rects_map()) {
//     for (auto pin_rect : pin_rect_list) {
//       BoostRect pinRect = DRCUtil::getBoostRect(pin_rect);
//       _layer_to_polygons_map[layerId] += pinRect;
//     }
//   }
// }

/**
 * @brief 将最小面积违规多边形导体的外包矩形作为违规矩形进行存储
 *
 * @param layerId 金属层
 * @param vialation_box 违规多边形导体的外接矩形
 * @param type 违规类型
 */
void RoutingAreaCheck::add_spot(int layerId, const DrcRectangle<int>& vialation_box, ViolationType type)
{
  // std::cout << "violation box lb :: (" << vialation_box.get_lb_x() << "," << vialation_box.get_lb_y() << ") , violation box rt :: (" <<
  // vialation_box.get_rt_x()
  //           << "," << vialation_box.get_rt_y() << ")" << std::endl;
  DrcSpot spot;

  DrcRect* drc_rect = new DrcRect();
  drc_rect->set_owner_type(RectOwnerType::kSpotMark);
  drc_rect->set_layer_id(layerId);
  drc_rect->set_rectangle(vialation_box);

  spot.set_violation_type(ViolationType::kRoutingArea);
  spot.add_spot_rect(drc_rect);
  _routing_layer_to_spots_map[layerId].emplace_back(spot);
}

// /**
//  * @brief 检查目标线网是否存在最小面积违规
//  *
//  * @param target_net
//  */
// void RoutingAreaCheck::checkRoutingArea(DrcNet* target_net)
// {
//   _layer_to_polygons_map.clear();
//   initLayerPolygonSet(target_net);
//   // checkRoutingArea();
//   checkRoutingArea(target_net->get_net_id());
// }

/**
 * @brief 初始化最小面积检查模块
 *
 * @param config
 * @param tech
 */
void RoutingAreaCheck::init(DrcConfig* config, Tech* tech, RegionQuery* region_query)
{
  _config = config;
  _tech = tech;
  _region_query = region_query;
}

/**
 * @brief 重置最小面积检查模块
 *
 */
void RoutingAreaCheck::reset()
{
  _layer_to_polygons_map.clear();

  for (auto& [LayerId, spot_list] : _routing_layer_to_spots_map) {
    for (auto& spot : spot_list) {
      spot.clearSpotRects();
    }
  }
  _routing_layer_to_spots_map.clear();
}

/**
 * @brief 获得最小面积违规数目
 *
 * @return int
 */
int RoutingAreaCheck::get_area_violation_num()
{
  int count = 0;
  for (auto& [layerId, spot_list] : _routing_layer_to_spots_map) {
    count += spot_list.size();
  }
  return count;
}

// /**
//  * @brief
//  * 跳过最小面积违规检查，如果最本线网的导体多边形与其它线网的导体相交导致短路，则跳过最小面积违规检查，因为在间距检查模块中已经进行了短路检查
//  *
//  * @param layerId 金属层Id
//  * @param netId 线网Id
//  * @param target_polygon 目标检查导体多边形
//  * @param query_box 以目标检查多边形的外接矩形为搜索区域
//  * @return true 跳过最小面积检查
//  * @return false 不跳过最小面积检查
//  */
// bool RoutingAreaCheck::skipCheck(int layerId, int netId, const BoostPolygon& target_polygon, const RTreeBox& query_box)
// {
//   //以目标检查导体多边形对象的外接矩形作为搜索区域进行搜索

//   std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
//   _region_query->queryInRoutingLayer(layerId, query_box, query_result);
//   std::vector<BoostPolygon> diff_net_poly_list;
//   //将搜索区域内与目标检查导体多边形不同net的矩形合成导体多边形
//   for (auto& rect_pair : query_result) {
//     DrcRect* result_rect = rect_pair.second;
//     if (result_rect->get_net_id() == netId) {
//       continue;
//     }
//     diff_net_poly_list += DRCUtil::getBoostRect(result_rect);
//   }
//   //遍历搜索区域内与目标矩形不同net的导体多边形，查看它们是否与目标导体多边形产生短路
//   for (auto diff_net_poly : diff_net_poly_list) {
//     auto overlap_poly = target_polygon & diff_net_poly;
//     if (bp::area(overlap_poly) != 0) {
//       return true;
//     }
//   }
//   return false;
// }

// /**
//  * @brief 检查目标net是否存在最小面积违规
//  *
//  * @param netId
//  */
// void RoutingAreaCheck::checkRoutingArea(int netId)
// {
//   for (auto& [layerId, polygon_set] : _layer_to_polygons_map) {
//     int require_area = _tech->getRoutingMinArea(layerId);
//     for (auto& polygon : polygon_set) {
//       int polygon_area = bp::area(polygon);
//       // BoostRect bounding_box;
//       // bp::extents(bounding_box, polygon);
//       // RTreeBox query_box = DRCUtil::getRTreeBox(bounding_box);
//       // if (skipCheck(layerId, netId, polygon, query_box)) {
//       //   continue;
//       // }
//       if (polygon_area < require_area) {
//         DrcRectangle violation_box = DRCUtil::getRectangleFromBoostRect(bounding_box);
//         add_spot(layerId, violation_box, ViolationType::kRoutingArea);
//       }
//     }
//   }
// }

// //////////*************************** interact with iRT ***************************///////////
// //目前没用到
// void RoutingAreaCheck::checkRoutingArea(const LayerNameToRTreeMap& layer_to_rects_tree_map)
// {
//   initLayerPolygonSet(layer_to_rects_tree_map);
//   checkRoutingArea();
// }

// void RoutingAreaCheck::initLayerPolygonSet(const LayerNameToRTreeMap& layer_to_rects_tree_map)
// {
//   for (auto& [layerName, rtree] : layer_to_rects_tree_map) {
//     int layerId = _tech->getLayerIdByLayerName(layerName);
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       DrcRect* target_rect = it->second;
//       _layer_to_polygons_map[layerId] += DRCUtil::getBoostRect(target_rect);
//     }
//   }
// }

// /////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////another way to check
// //目前没用到
// /**
//  * @brief 通过Boost合并每一金属层的矩形为导体多边形
//  *
//  */
// void RoutingAreaCheck::initLayerPolygonSet()
// {
//   for (auto& [layerId, rtree] : _region_query->get_layer_to_routing_rects_tree_map()) {
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       BoostRect layer_routing_rect = DRCUtil::getBoostRect(it->first);
//       _layer_to_polygons_map[layerId] += layer_routing_rect;
//     }
//   }
//   for (auto& [layerId, rtree] : _region_query->get_layer_to_fixed_rects_tree_map()) {
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       BoostRect layer_fixed_rect = DRCUtil::getBoostRect(it->first);
//       _layer_to_polygons_map[layerId] += layer_fixed_rect;
//     }
//   }
// }

// /**
//  * @brief 检查每一层中的各个导体多边形是否存在最小面积违规
//  *
//  */
// void RoutingAreaCheck::checkRoutingArea()
// {
//   for (auto& [layerId, polygon_set] : _layer_to_polygons_map) {
//     int require_area = _tech->getRoutingMinArea(layerId);
//     for (auto& polygon : polygon_set) {
//       int polygon_area = bp::area(polygon);
//       if (polygon_area < require_area) {
//         BoostRect bounding_box;
//         bp::extents(bounding_box, polygon);
//         DrcRectangle violation_box = DRCUtil::getRectangleFromBoostRect(bounding_box);
//         add_spot(layerId, violation_box, ViolationType::kRoutingArea);
//       }
//     }
//   }
// }

// /**
//  * @brief 另一种检查最小面积违规的方式，不按一条条net来，按层合并多边形并遍历每个多边形进行检查
//  *
//  */
// void RoutingAreaCheck::checkRoutingAreaLayerByLayer()
// {
//   _layer_to_polygons_map.clear();
//   initLayerPolygonSet();
//   checkRoutingArea();
// }

}  // namespace idrc