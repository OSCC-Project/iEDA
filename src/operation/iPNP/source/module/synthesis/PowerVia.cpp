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
/**
 * @file PowerVia.cpp
 * @author Jianrong Su
 * @brief 
 * @version 1.0
 * @date 2025-06-23
 */

#include "PowerVia.hh"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

  idb::IdbDesign* PowerVia::connectAllPowerLayers(GridManager& pnp_network, idb::IdbDesign* idb_design)
  {
    if (!idb_design) {
      LOG_INFO << "Error : Invalid IDB design object";
      return nullptr;
    }

    idb_design = connectNetworkLayers(pnp_network, PowerType::kVDD, idb_design);

    idb_design = connectNetworkLayers(pnp_network, PowerType::kVSS, idb_design);

    LOG_INFO << "Success : Connected all power layers";
    return idb_design;
  }

  idb::IdbDesign* PowerVia::connectM2M1Layer(idb::IdbDesign* idb_design)
  {
    idb_design = connect_M2_M1("VDD", idb_design);
    idb_design = connect_M2_M1("VSS", idb_design);

    LOG_INFO << "Success : Connected M2 and M1 layers";
    return idb_design;
  }

  idb::IdbDesign* PowerVia::connectNetworkLayers(GridManager& pnp_network, PowerType net_type, idb::IdbDesign* idb_design)
  {
    std::string net_name = (net_type == PowerType::kVDD) ? "VDD" : "VSS";

    auto power_layers = pnp_network.get_power_layers();
    int layer_count = pnp_network.get_layer_count();

    // Connect power layers
    for (int i = 0; i < layer_count - 1; i++) {
      std::string top_layer = "M" + std::to_string(power_layers[i]);
      std::string bottom_layer = "M" + std::to_string(power_layers[i + 1]);
      idb_design = connectLayers(net_name, top_layer, bottom_layer, idb_design);
    }

    // Connect power layer to M2 and M1 layer rows
    std::string bottom_power_layer = "M" + std::to_string(power_layers[layer_count - 1]);
    idb_design = connect_Layer_Row(net_name, bottom_power_layer, "M2", idb_design);
    
    LOG_INFO << "Success : Connected all layers for " << net_name;
    return idb_design;
  }

  int32_t PowerVia::transUnitDB(double value, idb::IdbDesign* idb_design)
  {
    if (!idb_design) return -1;
    auto idb_layout = idb_design->get_layout();
    return idb_layout != nullptr ? idb_layout->transUnitDB(value) : -1;
  }

  idb::IdbVia* PowerVia::findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, idb::IdbDesign* idb_design)
  {
    if (!idb_design) return nullptr;
    auto via_list = idb_design->get_via_list();

    // Via name format: cut_layer_name_widthxheight
    std::string via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design);

    // Search for existing via
    idb::IdbVia* via_find = via_list->find_via(via_name);

    // If not found, create a new via
    if (via_find == nullptr) {
      via_find = createVia(layer_cut, width_design, height_design, via_name, idb_design);
    }

    return via_find;
  }

  idb::IdbVia* PowerVia::createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name, idb::IdbDesign* idb_design)
  {
    if (!idb_design) return nullptr;
    auto via_list = idb_design->get_via_list();

    // Ensure via name is correctly formatted
    via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design);

    // Create via
    idb::IdbVia* via = via_list->createVia(via_name, layer_cut, width_design, height_design);

    // Debug information
    LOG_INFO << "Created via: " << via_name << ", via_list size: " << via_list->get_via_list().size();

    return via;
  }

  idb::IdbSpecialWireSegment* PowerVia::createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width,
    idb::IdbWireShapeType wire_shape_type,
    idb::IdbCoordinate<int32_t>* coord,
    idb::IdbVia* via)
  {
    // Create special wire segment
    idb::IdbSpecialWireSegment* segment_via = new idb::IdbSpecialWireSegment();

    // Set as via type
    segment_via->set_is_via(true);

    // Add coordinate point
    segment_via->add_point(coord->get_x(), coord->get_y());

    // Set layer information
    segment_via->set_layer(layer);

    // Set shape type
    segment_via->set_shape_type(idb::IdbWireShapeType::kStripe);

    // Mark as new layer
    segment_via->set_layer_as_new();

    // Set route width
    segment_via->set_route_width(0);

    // Copy via and set coordinates
    idb::IdbVia* via_new = segment_via->copy_via(via);
    if (via_new != nullptr) {
      via_new->set_coordinate(coord);
    }

    // Set bounding box
    segment_via->set_bounding_box();

    return segment_via;
  }

  bool PowerVia::getIntersectCoordinate(idb::IdbSpecialWireSegment* segment_top,
    idb::IdbSpecialWireSegment* segment_bottom,
    idb::IdbRect& intersection_rect)
  {
    // Get segment bounding boxes
    idb::IdbRect* top_bbox = segment_top->get_bounding_box();
    idb::IdbRect* bottom_bbox = segment_bottom->get_bounding_box();

    // Check if there is intersection
    if (!top_bbox->isIntersection(bottom_bbox)) {
      return false;
    }

    // Calculate intersection area
    int32_t ll_x = std::max(top_bbox->get_low_x(), bottom_bbox->get_low_x());
    int32_t ll_y = std::max(top_bbox->get_low_y(), bottom_bbox->get_low_y());
    int32_t ur_x = std::min(top_bbox->get_high_x(), bottom_bbox->get_high_x());
    int32_t ur_y = std::min(top_bbox->get_high_y(), bottom_bbox->get_high_y());

    // Set intersection rectangle
    intersection_rect.set_rect(ll_x, ll_y, ur_x, ur_y);

    return true;
  }

  bool PowerVia::addSingleVia(std::string net_name,
    std::string top_layer,
    std::string bottom_layer,
    double x, double y,
    int32_t width, int32_t height,
    idb::IdbDesign* idb_design)
  {
    if (!idb_design) return false;

    auto idb_layout = idb_design->get_layout();
    auto idb_layer_list = idb_layout->get_layers();
    auto idb_pdn_list = idb_design->get_special_net_list();

    // Find network
    idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
    if (net == nullptr) {
      LOG_INFO << "Error: Cannot find net " << net_name;
      return false;
    }

    // Get wire
    idb::IdbSpecialWire* wire = nullptr;
    if (net->get_wire_list()->get_num() > 0) {
      wire = net->get_wire_list()->find_wire(0);
    }
    else {
      wire = net->get_wire_list()->add_wire(nullptr);
    }

    if (wire == nullptr) {
      LOG_INFO << "Error: Cannot get wire for net " << net_name;
      return false;
    }

    // Find all cut layers
    std::vector<idb::IdbLayerCut*> cut_layer_list =
      idb_layer_list->find_cut_layer_list(top_layer, bottom_layer);

    if (cut_layer_list.empty()) {
      LOG_INFO << "Error: No cut layers found between " << top_layer << " and " << bottom_layer;
      return false;
    }

    // Convert coordinates to database units
    int32_t dbu_x = transUnitDB(x, idb_design);
    int32_t dbu_y = transUnitDB(y, idb_design);

    // Add via for each cut layer
    for (auto layer_cut : cut_layer_list) {
      if (!layer_cut->is_cut()) continue;

      // Find or create via
      idb::IdbVia* via = findVia(layer_cut, width, height, idb_design);
      if (via == nullptr) {
        LOG_INFO << "Error: Failed to create via for " << layer_cut->get_name();
        continue;
      }

      // Get via top layer
      idb::IdbLayer* via_top_layer = via->get_top_layer_shape().get_layer();

      // Create coordinate
      idb::IdbCoordinate<int32_t>* coord = new idb::IdbCoordinate<int32_t>(dbu_x, dbu_y);

      // Create via segment
      idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(
        via_top_layer, 0, idb::IdbWireShapeType::kStripe, coord, via);

      // Add to wire
      wire->add_segment(segment_via);
    }

    return true;
  }

  idb::IdbDesign* PowerVia::connectLayers(std::string net_name, std::string top_layer_name, std::string bottom_layer_name, idb::IdbDesign* idb_design)
  {
    if (!idb_design) {
      LOG_INFO << "Error : Invalid IDB design object";
      return nullptr;
    }

    auto idb_layout = idb_design->get_layout();
    auto idb_layer_list = idb_layout->get_layers();
    auto idb_pdn_list = idb_design->get_special_net_list();

    // Get layer information
    idb::IdbLayerRouting* layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(bottom_layer_name));
    idb::IdbLayerRouting* layer_top = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(top_layer_name));

    // Ensure layers exist and are different
    if (layer_bottom == nullptr || layer_top == nullptr || layer_bottom == layer_top) {
      LOG_INFO << "Error : layers not exist or same layer.";
      return nullptr;
    }

    // Ensure bottom layer is below top layer
    if (layer_top->get_order() < layer_bottom->get_order()) {
      std::swap(layer_top, layer_bottom);
    }

    // Don't support two layers with the same direction
    if ((layer_top->is_horizontal() && layer_bottom->is_horizontal()) ||
      (layer_top->is_vertical() && layer_bottom->is_vertical())) {
      LOG_INFO << "Error : layers have the same direction.";
      return nullptr;
    }

    // Get network
    idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
    if (net == nullptr) {
      LOG_INFO << "Error : can't find the net " << net_name;
      return nullptr;
    }

    // Get network wire list
    idb::IdbSpecialWireList* wire_list = net->get_wire_list();
    if (wire_list == nullptr) {
      LOG_INFO << "Error : not wire in Special net " << net_name;
      return nullptr;
    }

    std::vector<idb::IdbSpecialWireSegment*> segment_list_top;
    std::vector<idb::IdbSpecialWireSegment*> segment_list_bottom;
    idb::IdbSpecialWire* wire_top = nullptr;

    // Collect top and bottom layer segments
    for (idb::IdbSpecialWire* wire : wire_list->get_wire_list()) {
      for (idb::IdbSpecialWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_tripe() || segment->is_follow_pin()) {
          if (segment->get_layer()->compareLayer(layer_top)) {
            segment->set_bounding_box();
            segment_list_top.emplace_back(segment);
            wire_top = wire;
          }

          if (segment->get_layer()->compareLayer(layer_bottom)) {
            segment->set_bounding_box();
            segment_list_bottom.emplace_back(segment);
          }
        }
      }
    }
    
    // For each top layer segment
    for (idb::IdbSpecialWireSegment* segment_top : segment_list_top) {
      // For each bottom layer segment
      for (idb::IdbSpecialWireSegment* segment_bottom : segment_list_bottom) {
        // Calculate intersection area
        idb::IdbRect intersection_rect;
        if (getIntersectCoordinate(segment_top, segment_bottom, intersection_rect)) {
          // Debug information
          // LOG_INFO << "Found intersection at (" << intersection_rect.get_middle_point().get_x()
          //   << ", " << intersection_rect.get_middle_point().get_y()
          //   << ") with size " << intersection_rect.get_width() << "x" << intersection_rect.get_height();

          // Add via for each intermediate layer
          for (int32_t layer_order = layer_bottom->get_order();
            layer_order <= (layer_top->get_order() - 2);)
          {
            // Get cut layer
            idb::IdbLayerCut* layer_cut_find = dynamic_cast<idb::IdbLayerCut*>(
              idb_layer_list->find_layer_by_order(layer_order + 1));

            if (layer_cut_find == nullptr) {
              LOG_INFO << "Error : layer input illegal.";
              return nullptr;
            }

            // Find or create via
            idb::IdbVia* via_find = findVia(layer_cut_find,
              intersection_rect.get_width(),
              intersection_rect.get_height(),
              idb_design);

            if (via_find == nullptr) {
              LOG_INFO << "Error : can not find VIA matchs.";
              continue;
            }

            // Get via top layer
            idb::IdbLayer* layer_top_via = via_find->get_top_layer_shape().get_layer();

            // Create coordinate
            idb::IdbCoordinate<int32_t> middle = intersection_rect.get_middle_point();
            idb::IdbCoordinate<int32_t>* middle_ptr = new idb::IdbCoordinate<int32_t>(middle.get_x(), middle.get_y());

            // Create via segment
            idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(
              layer_top_via, 0, idb::IdbWireShapeType::kStripe, middle_ptr, via_find);

            // Add to wire
            wire_top->add_segment(segment_via);

            // Move to next cut layer
            layer_order += 2;
          }
        }
      }
    }

    LOG_INFO << "Success : connectLayers " << top_layer_name << " & " << bottom_layer_name;
    return idb_design;
  }

  
  idb::IdbDesign* PowerVia::connect_Layer_Row(std::string net_name, std::string top_layer_name, std::string bottom_layer_name, idb::IdbDesign* idb_design)
  {
    auto idb_layout = idb_design->get_layout();
    auto idb_layer_list = idb_layout->get_layers();
    auto idb_pdn_list = idb_design->get_special_net_list();

    // Get layer information
    idb::IdbLayerRouting* layer_M2 = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(bottom_layer_name));
    idb::IdbLayerRouting* layer_top = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(top_layer_name));
    idb::IdbLayerRouting* layer_M9 = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer("M9"));

    // Ensure layers exist and are different
    if (layer_M2 == nullptr || layer_top == nullptr || layer_M2 == layer_top) {
      LOG_INFO << "Error : layers not exist or same layer.";
      return nullptr;
    }

    // Ensure bottom layer is below top layer
    if (layer_top->get_order() < layer_M2->get_order()) {
      std::swap(layer_top, layer_M2);
    }

    // Don't support two layers with the same direction
    if ((layer_top->is_horizontal() && layer_M2->is_horizontal()) ||
      (layer_top->is_vertical() && layer_M2->is_vertical())) {
      LOG_INFO << "Error : layers have the same direction.";
      return nullptr;
    }

    // Get network
    idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
    if (net == nullptr) {
      LOG_INFO << "Error : can't find the net " << net_name;
      return nullptr;
    }

    // Get network wire list
    idb::IdbSpecialWireList* wire_list = net->get_wire_list();
    if (wire_list == nullptr) {
      LOG_INFO << "Error : not wire in Special net " << net_name;
      return nullptr;
    }

    std::vector<idb::IdbSpecialWireSegment*> segment_list_top;
    std::vector<idb::IdbSpecialWireSegment*> segment_list_bottom;
    idb::IdbSpecialWire* wire_top = nullptr;

    // Collect top and bottom layer segments
    for (idb::IdbSpecialWire* wire : wire_list->get_wire_list()) {
      for (idb::IdbSpecialWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_tripe() || segment->is_follow_pin()) {
          if (segment->get_layer()->compareLayer(layer_top)) {
            segment->set_bounding_box();
            segment_list_top.emplace_back(segment);
            // wire_top = wire;
          }
          if (segment->get_layer()->compareLayer(layer_M2)) {
            segment->set_bounding_box();
            segment_list_bottom.emplace_back(segment);
          }
        }
      }
    }

    // For each top layer segment
    for (idb::IdbSpecialWireSegment* segment_top : segment_list_top) {
      // For each bottom layer segment
      for (idb::IdbSpecialWireSegment* segment_bottom : segment_list_bottom) {
        // Calculate intersection area
        idb::IdbRect intersection_rect;
        if (getIntersectCoordinate(segment_top, segment_bottom, intersection_rect)) {
          
          int32_t layer_order_M2 = layer_M2->get_order();
          int32_t layer_order_top = layer_top->get_order();

          // Create coordinate
          idb::IdbCoordinate<int32_t> middle = intersection_rect.get_middle_point();
          idb::IdbCoordinate<int32_t>* middle_ptr = new idb::IdbCoordinate<int32_t>(middle.get_x(), middle.get_y());

          int wire_top_index = (layer_top->get_order() - layer_M9->get_order()) / (-2);
          
          // Process each cut layer
          for (int layer_cut_order = layer_order_top - 1;
            layer_cut_order > layer_order_M2 - 2; layer_cut_order -= 2) {

            wire_top = wire_list->find_wire(wire_top_index);

            // Get cut layer
            idb::IdbLayerCut* layer_cut = dynamic_cast<idb::IdbLayerCut*>(
              idb_layer_list->find_layer_by_order(layer_cut_order));

            if (layer_cut == nullptr) {
              LOG_INFO << "Error : layer input illegal.";
              continue;
            }

            // Find or create via
            idb::IdbVia* via_find = findVia(layer_cut,
              intersection_rect.get_width(),
              intersection_rect.get_height(),
              idb_design);
              
            if (via_find == nullptr) {
              LOG_INFO << "Error : can not find VIA matchs.";
              continue;
            }

            // Get via top layer
            idb::IdbLayer* layer_top_via = via_find->get_top_layer_shape().get_layer();

            // Create via segment
            idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(
              layer_top_via, 0, idb::IdbWireShapeType::kStripe, middle_ptr, via_find);

            // Add to wire
            wire_top->add_segment(segment_via);
            wire_top_index++;
          }
        }
      }
    }


    LOG_INFO << "Success : connectLayers " << top_layer_name << " & " << bottom_layer_name;
    return idb_design;
  }

  idb::IdbDesign* PowerVia::connect_M2_M1(std::string net_name, idb::IdbDesign* idb_design)
  {
    auto idb_layout = idb_design->get_layout();
    auto idb_layer_list = idb_layout->get_layers();
    auto idb_pdn_list = idb_design->get_special_net_list();
    auto idb_via_list = idb_design->get_via_list();

    // Get layer information
    idb::IdbLayerRouting* layer_M3 = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer("M3"));
    idb::IdbLayerRouting* layer_M2 = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer("M2"));
    idb::IdbLayerRouting* layer_M1 = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer("M1"));
    idb::IdbLayerCut* layer_via1 = dynamic_cast<idb::IdbLayerCut*>(
      idb_layer_list->find_layer("VIA1"));
    idb::IdbLayerCut* layer_via2 = dynamic_cast<idb::IdbLayerCut*>(
      idb_layer_list->find_layer("VIA2"));

    if (!layer_M3 || !layer_M2 || !layer_M1 || !layer_via1 || !layer_via2) {
      LOG_INFO << "Error: Cannot find required layers";
      return nullptr;
    }

    // Get network
    idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
    if (net == nullptr) {
      LOG_INFO << "Error : can't find the net " << net_name;
      return nullptr;
    }

    // Get network wire list
    idb::IdbSpecialWireList* wire_list = net->get_wire_list();
    if (wire_list == nullptr) {
      LOG_INFO << "Error : not wire in Special net " << net_name;
      return nullptr;
    }

    // 首先检查是否已经存在VIAGEN12_RECT_1 via
    idb::IdbVia* m2_m1_via = idb_via_list->find_via("VIAGEN12_RECT_1");
    
    // 如果不存在，则需要创建
    if (m2_m1_via == nullptr) {
      LOG_INFO << "VIAGEN12_RECT_1 not found, creating it...";
      
      // 查找M3-M2 via作为模板
      idb::IdbVia* m3_m2_via = nullptr;
      
      for (auto via : idb_via_list->get_via_list()) {
        auto cut_layer_shape = via->get_cut_layer_shape();
        auto cut_layer = cut_layer_shape.get_layer();
        if (cut_layer->compareLayer(layer_via2) && via->get_name().find("VIAGEN23") != std::string::npos) {
          m3_m2_via = via;
          LOG_INFO << "Found M3-M2 via : " << via->get_name();
          break;
        }
      }
      
      if (m3_m2_via) {
        // clone via
        m2_m1_via = m3_m2_via->clone();

        // set via master
        idb::IdbViaMaster* m2_m1_via_master = m3_m2_via->get_instance()->clone();

        // set via master generate
        idb::IdbViaMasterGenerate* m2_m1_via_master_generate = m2_m1_via_master->get_master_generate()->clone();
        m2_m1_via_master_generate->set_rule_name("VIAGEN12_RECT_1");
        m2_m1_via_master_generate->set_layer_bottom(layer_M1);
        m2_m1_via_master_generate->set_layer_top(layer_M2);
        m2_m1_via_master_generate->set_layer_cut(layer_via1);

        // set rule generate
        idb::IdbViaRuleGenerate* m2_m1_via_rule_generate = m2_m1_via_master_generate->get_rule_generate();
        m2_m1_via_rule_generate->set_name("VIAGEN12_RECT_1");
        m2_m1_via_rule_generate->set_layer_bottom(layer_M1);
        m2_m1_via_rule_generate->set_layer_top(layer_M2);
        m2_m1_via_rule_generate->set_layer_cut(layer_via1);

        m2_m1_via_master_generate->set_rule_generate(m2_m1_via_rule_generate);
        m2_m1_via_master->set_master_generate(m2_m1_via_master_generate);
        m2_m1_via->set_instance(m2_m1_via_master);

        // set via name
        m2_m1_via->set_name("VIAGEN12_RECT_1");

        idb_via_list->add_via(m2_m1_via);

        LOG_INFO << "Created new M2-M1 via: VIAGEN12_RECT_1 based on " << m3_m2_via->get_name();
      } else {
        LOG_INFO << "Error: Cannot find M3-M2 via template";
        return idb_design;
      }
    } else {
      LOG_INFO << "Using existing VIAGEN12_RECT_1 via for " << net_name;
    }

    // 无论是新创建还是已存在，都使用m2_m1_via添加到网络中
    for (idb::IdbSpecialWire* wire : wire_list->get_wire_list()) {
      auto segment_list = wire->get_segment_list();

      for (idb::IdbSpecialWireSegment* segment : segment_list) {
        if (segment->is_via() && segment->get_layer()->compareLayer(layer_M3)) {
          // 在M2层添加通孔
          idb::IdbSpecialWireSegment* new_segment = new idb::IdbSpecialWireSegment();
          new_segment->set_layer(layer_M2);
          new_segment->set_is_via(true);
          new_segment->set_route_width(0);

          // 使用M2-M1 via
          new_segment->set_via(m2_m1_via);

          // 复制坐标点
          if (!segment->get_point_list().empty()) {
            new_segment->add_point(segment->get_point_list()[0]->get_x(), segment->get_point_list()[0]->get_y());
          }

          // 添加到wire中
          wire->add_segment(new_segment);
        }
      }
    }

    return idb_design;
  }
  
}  // namespace ipnp 