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
#include "IDRWrapper.h"

namespace idrc {

/**
 * @brief  initialize all fix shape in drc_design
 *
 * @param dr_design
 */
// void IDRWrapper::inputAndInitFixedShapes(idr::Design* dr_design)
// {
//   double start = 0.0;
//   double end = 0.0;
//   start = DRCCOMUtil::microtime();

//   set_dr_design(dr_design);
//   wrapPinListAndBlockageList();

//   end = DRCCOMUtil::microtime();
//   std::cout << "[IDRWrapper Info] \033[1;32mTotal elapsed time:" << (end - start) << "s \033[0m\n";
// }

// void IDRWrapper::wrapPinListAndBlockageList()
// {
//   wrapNetPinList();
//   wrapBlockageList();
// }
/**
 * @brief initialize all routing shape in drc_design from region router
 *
 * @param sub_net_list
 */
// void IDRWrapper::inputAndInitRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list)
// {
//   for (auto& sub_net : sub_net_list) {
//     DrcNet* drc_net = get_drc_net(sub_net.get_rr_net_idx());
//     wrapSegmentsAndVias(drc_net, sub_net);
//   }
// }

// void IDRWrapper::wrapSegmentsAndVias(DrcNet* drc_net, idr::RRSubNet& sub_net)
// {
//   // std::vector<idr::SpaceSegment<idr::RRPoint>> dr_space_segment_list = sub_net.get_best_routing_seg_list();
//   std::vector<idr::SpaceSegment<idr::RRPoint>> dr_space_segment_list = sub_net.get_curr_seg_list();
//   for (auto& dr_space_segment : dr_space_segment_list) {
//     if (dr_space_segment.is_via()) {
//       wrapNetVia(dr_space_segment, drc_net);
//     } else if (dr_space_segment.is_wiring()) {
//       wrapNetSegment(dr_space_segment, drc_net);
//     }
//   }
// }

// void IDRWrapper::inputAndInitCurrentRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list)
// {
//   for (auto& sub_net : sub_net_list) {
//     DrcNet* drc_net = get_drc_net(sub_net.get_rr_net_idx());
//     std::vector<idr::SpaceSegment<idr::RRPoint>> dr_space_segment_list = sub_net.get_curr_seg_list();
//     wrapSegmentsAndVias(drc_net, dr_space_segment_list);
//   }
// }

// void IDRWrapper::inputAndInitBestRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list)
// {
//   for (auto& sub_net : sub_net_list) {
//     DrcNet* drc_net = get_drc_net(sub_net.get_rr_net_idx());
//     std::vector<idr::SpaceSegment<idr::RRPoint>> dr_space_segment_list = sub_net.get_best_routing_seg_list();
//     wrapSegmentsAndVias(drc_net, dr_space_segment_list);
//   }
// }

// void IDRWrapper::inputAndInitRoutingShapesInRRNetList(std::vector<idr::RRNet>& rr_net_list)
// {
//   for (auto& rr_net : rr_net_list) {
//     DrcNet* drc_net = get_drc_net(rr_net.get_net_idx());
//     std::vector<idr::SpaceSegment<idr::RRPoint>> dr_space_segment_list = rr_net.get_routing_seg_list();
//     wrapSegmentsAndVias(drc_net, dr_space_segment_list);
//   }
// }

// void IDRWrapper::wrapSegmentsAndVias(DrcNet* drc_net, std::vector<idr::SpaceSegment<idr::RRPoint>>& dr_space_segment_list)
// {
//   for (auto& dr_space_segment : dr_space_segment_list) {
//     if (dr_space_segment.is_via()) {
//       wrapNetVia(dr_space_segment, drc_net);
//     } else if (dr_space_segment.is_wiring()) {
//       wrapNetSegment(dr_space_segment, drc_net);
//     }
//   }
// }

////wrap rect
// void IDRWrapper::wrapRect(DrcRectangle<int>& rect, idr::Rectangle<int>& dr_rect)
// {
//   rect.set_lb(dr_rect.get_lb_x(), dr_rect.get_lb_y());
//   rect.set_rt(dr_rect.get_rt_x(), dr_rect.get_rt_y());
// }
// void IDRWrapper::wrapRect(DrcRect* drc_rect, idr::Rectangle<int>& dr_rect)
// {
//   if (drc_rect == nullptr) {
//     std::cout << "[IDRWrapper error:drc_rect is null]" << std::endl;
//     return;
//   }
//   drc_rect->set_lb(dr_rect.get_lb_x(), dr_rect.get_lb_y());
//   drc_rect->set_rt(dr_rect.get_rt_x(), dr_rect.get_rt_y());
// }

/**
 * @brief wrap Blockage List
 *
 */
// void IDRWrapper::wrapBlockageList()
// {
//   DrcDesign* drc_design = get_drc_design();
//   std::vector<idr::Blockage> _idr_blockage_list = _dr_design->get_blockage_list();
//   // std::cout<<"block_list.size is :: "<<_idr_blockage_list.size()<<std::endl;
//   for (auto& dr_block : _idr_blockage_list) {
//     DrcRect* drc_block = new DrcRect();
//     wrapBlockage(dr_block, drc_block);
//     int layerId = drc_block->get_layer_id();
//     drc_design->add_blockage(layerId, drc_block);
//     _region_query->add_fixed_rect_to_rtree(layerId, drc_block);
//   }
// }

// void IDRWrapper::wrapBlockage(idr::Blockage& dr_block, DrcRect* drc_block)
// {
//   int layerId = dr_block.get_layer_idx() + 1;

//   drc_block->set_owner_type(RectOwnerType::kBlockage);
//   drc_block->set_layer_id(layerId);
//   drc_block->set_is_fixed(true);

//   wrapRect(drc_block, dr_block.get_region());
// }

/**
 * @brief wrap Net Pin List
 *
 */
// void IDRWrapper::wrapNetPinList()
// {
//   std::vector<idr::Net> _idr_net_list = _dr_design->get_net_list();
//   for (auto& dr_net : _idr_net_list) {
//     DrcNet* drc_net = get_drc_net(dr_net.get_net_idx());
//     wrapNetPinList(dr_net, drc_net);
//   }
// }

// void IDRWrapper::wrapNetPinList(idr::Net& dr_net, DrcNet* drc_net)
// {
//   std::vector<idr::Pin> dr_pin_list = dr_net.get_pin_list();
//   for (auto& dr_pin : dr_pin_list) {
//     std::vector<idr::Port> dr_port_list = dr_pin.get_port_list();
//     for (auto& dr_port : dr_port_list) {
//       int layer_id = dr_port.get_layer_idx() + 1;
//       for (auto& dr_rect : dr_port.get_port_shape()) {
//         DrcRectangle<int> rect;
//         wrapRect(rect, dr_rect);
//         DrcRect* pin_rect = new DrcRect(layer_id, rect);
//         pin_rect->set_net_id(drc_net->get_net_id());
//         pin_rect->set_owner_type(RectOwnerType::kPin);
//         pin_rect->set_is_fixed(true);

//         drc_net->add_pin_rect(layer_id, pin_rect);
//         _region_query->add_fixed_rect_to_rtree(layer_id, pin_rect);
//       }
//     }
//   }
// }

/**
 * @brief transform dr_space_segment to drc rect
 *
 * @param dr_space_segment_list
 * @param drc_net
 */
// void IDRWrapper::wrapNetVia(idr::SpaceSegment<idr::RRPoint>& dr_space_segment, DrcNet* drc_net)
// {
//   Tech* tech = get_tech();
//   // via_idx should match with in tech
//   int dr_via_idx = dr_space_segment.get_via_lib_idx();
//   idr::RRPoint first_point = dr_space_segment.get_segment().get_first();

//   idr::Coordinate<int> dr_via_coor = first_point.get_coord();
//   // via_idx in dr should match with in tech
//   DrcVia* via = tech->findViaByIdx(dr_via_idx);
//   // bottom
//   DrcEnclosure bottom_enclosure = via->get_bottom_enclosure();
//   DrcRect* drc_via_bottom = getViaMetalRect(dr_via_coor.get_x(), dr_via_coor.get_y(), bottom_enclosure);
//   if (drc_via_bottom != nullptr) {
//     drc_via_bottom->set_net_id(drc_net->get_net_id());
//     drc_via_bottom->set_via_idx(via->get_via_idx());
//     drc_net->add_routing_rect(drc_via_bottom->get_layer_id(), drc_via_bottom);
//     _region_query->add_routing_rect_to_rtree(drc_via_bottom->get_layer_id(), drc_via_bottom);
//   }
// top
//   DrcEnclosure top_enclosure = via->get_top_enclosure();
//   DrcRect* drc_via_top = getViaMetalRect(dr_via_coor.get_x(), dr_via_coor.get_y(), top_enclosure);
//   if (drc_via_top != nullptr) {
//     drc_via_top->set_net_id(drc_net->get_net_id());
//     drc_via_top->set_via_idx(via->get_via_idx());
//     drc_net->add_routing_rect(drc_via_top->get_layer_id(), drc_via_top);
//     _region_query->add_routing_rect_to_rtree(drc_via_top->get_layer_id(), drc_via_top);
//   }
// }

// DrcRect* IDRWrapper::getViaMetalRect(int center_x, int center_y, DrcEnclosure& enclosure)
// {
//   int layerId = enclosure.get_layer_idx();
//   int lb_x = enclosure.get_shape().get_lb_x() + center_x;
//   int lb_y = enclosure.get_shape().get_lb_y() + center_y;
//   int rt_x = enclosure.get_shape().get_rt_x() + center_x;
//   int rt_y = enclosure.get_shape().get_rt_y() + center_y;
//   DrcRectangle<int> rectangle(lb_x, lb_y, rt_x, rt_y);
//   // Detection and de duplication
//   if (_region_query->isExistingRectangleInRoutingRTree(layerId, rectangle)) {
//     // std::cout << "[DRC IDRWrapper Info]:Duplicate Rectangle From Via" << std::endl;
//     return nullptr;
//   }

//   DrcRect* drc_via_metal_rect = new DrcRect(layerId, rectangle);
//   drc_via_metal_rect->set_owner_type(RectOwnerType::kViaMetal);
//   drc_via_metal_rect->set_is_fixed(false);
//   return drc_via_metal_rect;
// }

// /**
//  * @brief transform dr_space_segment to drc rect
//  *
//  * @param dr_space_segment
//  * @param drc_net
//  */
// void IDRWrapper::wrapNetSegment(idr::SpaceSegment<idr::RRPoint>& dr_space_segment, DrcNet* drc_net)
// {
//   idr::Segment<idr::RRPoint> dr_segment = dr_space_segment.get_segment();
//   idr::RRPoint first_point = dr_segment.get_first();
//   idr::RRPoint second_point = dr_segment.get_second();
//   // segment begin and end
//   int begin_x = std::min(first_point.get_coord().get_x(), second_point.get_coord().get_x());
//   int begin_y = std::min(first_point.get_coord().get_y(), second_point.get_coord().get_y());
//   int end_x = std::max(first_point.get_coord().get_x(), second_point.get_coord().get_x());
//   int end_y = std::max(first_point.get_coord().get_y(), second_point.get_coord().get_y());

//   int layerId = first_point.get_layer_idx() + 1;
//   int wire_width = dr_space_segment.get_wire_width();
//   DrcRectangle<int> rectangle(begin_x - wire_width / 2, begin_y - wire_width / 2, end_x + wire_width / 2, end_y + wire_width / 2);

//   // Detection and de duplication
//   if (_region_query->isExistingRectangleInRoutingRTree(layerId, rectangle)) {
//     // std::cout << "[DRC IDRWrapper Info]:Duplicate Rectangle From Segment" << std::endl;
//     return;
//   }

//   DrcRect* segment_rect = new DrcRect(layerId, rectangle);
//   segment_rect->set_net_id(drc_net->get_net_id());
//   segment_rect->set_owner_type(RectOwnerType::kSegment);
//   segment_rect->set_is_fixed(false);
//   drc_net->add_routing_rect(layerId, segment_rect);
//   _region_query->add_routing_rect_to_rtree(layerId, segment_rect);
// }

// /////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////for region routing

// DrcNet* IDRWrapper::get_drc_net(int netId)
// {
//   DrcNet* drc_net = nullptr;
//   auto it = _id_to_net.find(netId);
//   if (it == _id_to_net.end()) {
//     DrcDesign* drc_design = get_drc_design();
//     drc_net = drc_design->add_drc_net();
//     drc_net->set_net_id(netId);
//     _id_to_net[netId] = drc_net;
//   } else {
//     drc_net = it->second;
//   }
//   return drc_net;
// }

/**
 * @brief wrapRoutingLayerList
 *
 */
// void IDRWrapper::wrapRoutingLayerList()
// {
//   Tech* tech = get_tech();
//   std::vector<idr::RoutingLayer> _dr_routing_layer_list = _dr_design->get_routing_layer_list();
//   for (auto& dr_routing_layer : _dr_routing_layer_list) {
//     DrcRoutingLayer* drc_routing_layer = tech->add_routing_layer();
//     wrapRoutingLayer(dr_routing_layer, drc_routing_layer);
//   }
// }

// void IDRWrapper::wrapRoutingLayer(idr::RoutingLayer& dr_routing_layer, DrcRoutingLayer* drc_routing_layer)
// {
//   drc_routing_layer->set_name(dr_routing_layer.get_layer_name());
//   drc_routing_layer->set_layer_id(dr_routing_layer.get_layer_idx() + 1);
//   drc_routing_layer->set_layer_type(LayerType::kRouting);
//   drc_routing_layer->set_direction(dr_routing_layer.isRoutingH() ? LayerDirection::kHorizontal : LayerDirection::kVertical);
//   drc_routing_layer->set_min_width(dr_routing_layer.get_min_width());

//   std::vector<idr::Spacing> dr_spacing_list = dr_routing_layer.get_spacing_list();
//   drc_routing_layer->set_min_spacing(dr_spacing_list.front().get_min_spacing());
//   for (size_t i = 1; i < dr_spacing_list.size(); ++i) {
//     SpacingRangeRule* spacing_range_rule = drc_routing_layer->add_spacing_range_rule();
//     spacing_range_rule->set_min_width(dr_spacing_list[i].get_min_range());
//     spacing_range_rule->set_max_width(dr_spacing_list[i].get_max_range());
//     spacing_range_rule->set_spacing(dr_spacing_list[i].get_min_spacing());
//   }
// }

/**
 * @brief wrap Via Lib
 *
 */
// void IDRWrapper::wrapViaLib()
// {
//   Tech* tech = get_tech();
//   std::vector<idr::Via> dr_via_list = _dr_design->get_via_lib();

//   for (auto& dr_via : dr_via_list) {
//     Via* via = tech->add_via();
//     wrapVia(dr_via, via);
//   }
// }

// void IDRWrapper::wrapVia(idr::Via& dr_via, Via* via)
// {
//   if (via == nullptr) {
//     std::cout << "[IDRWrapper Error] via is empty!" << std::endl;
//     exit(1);
//   }

//   via->set_via_name(dr_via.get_via_name());
//   /// set index
//   via->set_via_idx(dr_via.get_via_idx());

//   // set layer

//   /// top
//   idr::Enclosure dr_top_enclosure = dr_via.get_top_enclosure();

//   Enclosure top_enclosure;
//   top_enclosure.set_layer_idx(dr_top_enclosure.get_layer_idx() + 1);

//   Rectangle<int> rect_top;
//   wrapRect(rect_top, dr_top_enclosure.get_shape());
//   top_enclosure.set_shape(rect_top);
//   via->set_top_enclosure(top_enclosure);

//   /// bottom
//   idr::Enclosure dr_bottom_enclosure = dr_via.get_bottom_enclosure();

//   Enclosure bottom_enclosure;
//   bottom_enclosure.set_layer_idx(dr_bottom_enclosure.get_layer_idx() + 1);

//   Rectangle<int> rect_bottom;
//   wrapRect(rect_bottom, dr_bottom_enclosure.get_shape());
//   bottom_enclosure.set_shape(rect_bottom);
//   via->set_bottom_enclosure(bottom_enclosure);
// }

}  // namespace idrc