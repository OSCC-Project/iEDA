// #include "CutSpacingCheck.hpp"
// #include "DRC.h"
// #include "DrcConfig.h"
// #include "DrcConfigurator.h"
// #include "DrcDesign.h"
// #include "DrcIDBWrapper.h"
// #include "EnclosedAreaCheck.h"
// #include "EnclosureCheck.hpp"
// #include "IDRWrapper.h"
// #include "MultiPatterning.h"
// #include "RegionQuery.h"
// #include "RoutingAreaCheck.h"
// #include "RoutingSpacingCheck.h"
// #include "RoutingWidthCheck.h"
// #include "SpotParser.h"
// #include "Tech.h"

// namespace idrc {

// void DRC::initDesignBlockPolygon()
// {
//   std::vector<PolygonWithHoles> poly_with_holes_list;
//   for (auto& [layerId, polyset] : _drc_design->get_blockage_polygon_set_list()) {
//     poly_with_holes_list.clear();
//     polyset.get(poly_with_holes_list);
//     for (auto& poly_with_holes : poly_with_holes_list) {
//       DrcPolygon* polygon = _drc_design->add_blockage_polygon(layerId, poly_with_holes);
//       bindRectangleToPolygon(polygon);
//     }
//   }
//   _drc_design->clear_blockage_polygon_set_list();
// }

// void DRC::initNetsMergePolygon()
// {
//   for (auto& net : _drc_design->get_drc_net_list()) {
//     initNetMergePolygon(net);
//     net->clear_layer_to_routing_polygon_set();
//     net->clear_layer_to_pin_polygon_set();
//   }
// }

// void DRC::initNetMergePolygon(DrcNet* net)
// {
//   std::set<int> layer_id_list = net->get_layer_id_list();
//   std::vector<PolygonWithHoles> poly_with_holes_list;

//   for (int layerId : layer_id_list) {
//     poly_with_holes_list.clear();
//     PolygonSet poly_set;
//     // merge
//     poly_set += net->get_routing_polygon_set(layerId);
//     poly_set += net->get_pin_polygon_set(layerId);
//     poly_set.get(poly_with_holes_list);
//     for (auto& poly_with_holes : poly_with_holes_list) {
//       DrcPolygon* merge_poly = net->add_merge_polygon(layerId, poly_with_holes);
//       merge_poly->set_net_id(net->get_net_id());
//       bindRectangleToPolygon(merge_poly);
//       // initNetMergePolyEdge(merge_poly);
//     }
//   }
// }

// void DRC::bindRectangleToPolygon(DrcPolygon* polygon)
// {
//   PolygonWithHoles poly_with_hole = polygon->get_polygon();
//   PolygonSet polyset1;
//   polyset1 += poly_with_hole;
//   /////////////////////////////////////////////////
//   BoostRect query_rect;
//   bp::extents(query_rect, poly_with_hole);
//   RTreeBox query_box = DRCUtil::getRTreeBox(query_rect);
//   std::vector<std::pair<RTreeBox, DrcRect*>> query_result;
//   _region_query->queryInRoutingLayer(polygon->get_layer_id(), query_box, query_result);
//   for (auto& [box, rect] : query_result) {
//     if (rect->get_net_id() != polygon->get_net_id()) {
//       continue;
//     }
//     BoostRect result_rect = DRCUtil::getBoostRect(rect);
//     PolygonSet polyset2;
//     polyset2 += result_rect;
//     // can`t 1 interact 2
//     if (!(polyset2.interact(polyset1).empty())) {
//       rect->set_owner_polygon(polygon);
//     }
//   }
// }

// void DRC::getObjectNum()
// {
//   int blockage_num = 0;
//   for (auto& [layerId, rectList] : _drc_design->get_layer_to_blockage_list()) {
//     blockage_num += rectList.size();
//   }
//   int routing_shape_num = 0;
//   int pin_shape_num = 0;
//   for (auto& net : _drc_design->get_drc_net_list()) {
//     for (auto& [layerId, shape_list] : net->get_layer_to_routing_rects_map()) {
//       routing_shape_num += shape_list.size();
//     }
//     for (auto& [layerId, shape_list] : net->get_layer_to_pin_rects_map()) {
//       pin_shape_num += shape_list.size();
//     }
//   }
//   std::cout << "Pin num : " << pin_shape_num << std::endl;
//   std::cout << "via and segment num : " << routing_shape_num << std::endl;
//   std::cout << "blockage num : " << blockage_num << std::endl;
// }

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////not use now
// void DRC::initNetMergePolyEdgeOuter(DrcPolygon* polygon, std::set<int>& x_value_list, std::set<int>& y_value_list)
// {
//   PolygonWithHoles poly_with_hole = polygon->get_polygon();
//   // skip the first point
//   auto outerIt = poly_with_hole.begin();
//   BoostPoint first_point((*outerIt).x(), (*outerIt).y());
//   BoostPoint begin_temp_point((*outerIt).x(), (*outerIt).y());
//   x_value_list.insert((*outerIt).x());
//   y_value_list.insert((*outerIt).y());
//   BoostPoint end_temp_point;
//   ++outerIt;
//   for (; outerIt != poly_with_hole.end(); ++outerIt) {
//     end_temp_point.x((*outerIt).x());
//     end_temp_point.y((*outerIt).y());
//     x_value_list.insert((*outerIt).x());
//     y_value_list.insert((*outerIt).y());
//     BoostSegment segment(begin_temp_point, end_temp_point);
//     addSegmentToDrcPolygon(segment, polygon);

//     begin_temp_point = end_temp_point;
//   }
//   BoostSegment last_segment(begin_temp_point, first_point);
//   addSegmentToDrcPolygon(last_segment, polygon);
// }

// void DRC::initNetMergePolyEdgeInner(DrcPolygon* polygon, std::set<int>& x_value_list, std::set<int>& y_value_list)
// {
//   // Todo
// }

// void DRC::initNetMergePolyEdge(DrcPolygon* polygon)
// {
//   std::set<int> x_value_list;
//   std::set<int> y_value_list;

//   initNetMergePolyEdgeOuter(polygon, x_value_list, y_value_list);
//   // Todo initNetMergePolyEdgeInner
//   // Todo initEdgeWidth
// }

// void DRC::addSegmentToDrcPolygon(const BoostSegment& segment, DrcPolygon* polygon)
// {
//   // std::unique_ptr<DrcEdge> drcEdge = std::make_unique<DrcEdge>(segment);
//   // DrcEdge* drc_edge = drcEdge.get();
//   // Todo set edge direct!!!!!!!!
//   // drc_edge->set_owner_polygon(polygon);
//   // polygon->add_polygon_edge(drcEdge);
//   // _region_query->add_routing_edge_to_rtree(polygon->get_layer_id(), drc_edge);
// }

// void DRC::initConflictGraphByPolygon()
// {
//   _routing_sapcing_check->initConflictGraphByPolygon();
// }
// }  // namespace idrc