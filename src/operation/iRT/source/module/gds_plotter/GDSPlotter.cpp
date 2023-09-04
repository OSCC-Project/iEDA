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
#include "GDSPlotter.hpp"

#include "GPGraphType.hpp"
#include "GPLYPLayer.hpp"
#include "GPLayoutType.hpp"
#include "MTree.hpp"
#include "Monitor.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void GDSPlotter::initInst()
{
  if (_gp_instance == nullptr) {
    _gp_instance = new GDSPlotter();
  }
}

GDSPlotter& GDSPlotter::getInst()
{
  if (_gp_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_gp_instance;
}

void GDSPlotter::destroyInst()
{
  if (_gp_instance != nullptr) {
    delete _gp_instance;
    _gp_instance = nullptr;
  }
}

// function

void GDSPlotter::plot(Net& net, Stage stage, bool add_layout, bool need_clipping)
{
  std::vector<Net> net_list = {net};
  plot(net_list, stage, add_layout, need_clipping);
}

void GDSPlotter::plot(std::vector<Net>& net_list, Stage stage, bool add_layout, bool need_clipping)
{
  std::string gp_temp_directory_path = DM_INST.getConfig().gp_temp_directory_path;

  GPGDS gp_gds;
  if (stage == Stage::kResourceAllocator) {
    addCostMap(gp_gds, net_list);
    add_layout = false;
    need_clipping = false;
  } else {
    addNetList(gp_gds, net_list, stage);
  }
  std::string gds_file_path = RTUtil::getString(gp_temp_directory_path, GetStageName()(stage), ".gds");
  plot(gp_gds, gds_file_path, add_layout, need_clipping);
}

void GDSPlotter::plot(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping)
{
  plotGDS(gp_gds, gds_file_path, add_layout, need_clipping);
}

irt_int GDSPlotter::getGDSIdxByRouting(irt_int routing_layer_idx)
{
  irt_int gds_layer_idx = 0;
  if (RTUtil::exist(_routing_layer_gds_map, routing_layer_idx)) {
    gds_layer_idx = _routing_layer_gds_map[routing_layer_idx];
  } else {
    LOG_INST.warning(Loc::current(), "The routing_layer_idx '", routing_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

irt_int GDSPlotter::getGDSIdxByCut(irt_int cut_layer_idx)
{
  irt_int gds_layer_idx = 0;
  if (RTUtil::exist(_cut_layer_gds_map, cut_layer_idx)) {
    gds_layer_idx = _cut_layer_gds_map[cut_layer_idx];
  } else {
    LOG_INST.warning(Loc::current(), "The cut_layer_idx '", cut_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

// private

GDSPlotter* GDSPlotter::_gp_instance = nullptr;

void GDSPlotter::init()
{
  buildGDSLayerMap();
  buildLayoutLypFile();
  buildGraphLypFile();
}

void GDSPlotter::buildGDSLayerMap()
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  std::map<irt_int, irt_int> order_gds_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    order_gds_map[routing_layer.get_layer_order()] = -1;
  }
  for (CutLayer& cut_layer : cut_layer_list) {
    order_gds_map[cut_layer.get_layer_order()] = -1;
  }
  // 0为die 最后一个为GCell 中间为cut+routing
  irt_int gds_layer_idx = 1;
  for (auto it = order_gds_map.begin(); it != order_gds_map.end(); it++) {
    it->second = gds_layer_idx++;
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int gds_layer_idx = order_gds_map[routing_layer.get_layer_order()];
    _routing_layer_gds_map[routing_layer.get_layer_idx()] = gds_layer_idx;
    _gds_routing_layer_map[gds_layer_idx] = routing_layer.get_layer_idx();
  }
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int gds_layer_idx = order_gds_map[cut_layer.get_layer_order()];
    _cut_layer_gds_map[cut_layer.get_layer_idx()] = gds_layer_idx;
    _gds_cut_layer_map[gds_layer_idx] = cut_layer.get_layer_idx();
  }
}

void GDSPlotter::buildLayoutLypFile()
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::string gp_temp_directory_path = DM_INST.getConfig().gp_temp_directory_path;

  std::vector<std::string> color_list = {"#ff9d9d", "#ff80a8", "#c080ff", "#9580ff", "#8086ff", "#80a8ff", "#ff0000", "#ff0080", "#ff00ff",
                                         "#8000ff", "#0000ff", "#0080ff", "#800000", "#800057", "#800080", "#500080", "#000080", "#004080",
                                         "#80fffb", "#80ff8d", "#afff80", "#f3ff80", "#ffc280", "#ffa080", "#00ffff", "#01ff6b", "#91ff00",
                                         "#ddff00", "#ffae00", "#ff8000", "#008080", "#008050", "#008000", "#508000", "#808000", "#805000"};
  std::vector<std::string> pattern_list = {"I5", "I9"};

  std::map<GPLayoutType, bool> routing_data_type_visible_map
      = {{GPLayoutType::kText, false},   {GPLayoutType::kPinShape, true},     {GPLayoutType::kAccessPoint, true},
         {GPLayoutType::kGuide, false},  {GPLayoutType::kPreferTrack, false}, {GPLayoutType::kNonpreferTrack, false},
         {GPLayoutType::kWire, true},    {GPLayoutType::kEnclosure, true},    {GPLayoutType::kPatch, true},
         {GPLayoutType::kBlockage, true}};
  std::map<GPLayoutType, bool> cut_data_type_visible_map
      = {{GPLayoutType::kText, false}, {GPLayoutType::kCut, true}, {GPLayoutType::kPatch, true}, {GPLayoutType::kBlockage, true}};

  // 0为die 最后一个为gcell 中间为cut+routing
  irt_int gds_layer_size = 2 + static_cast<irt_int>(_gds_routing_layer_map.size() + _gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (irt_int gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    std::string color = color_list[gds_layer_idx % color_list.size()];
    std::string pattern = pattern_list[gds_layer_idx % pattern_list.size()];

    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(color, pattern, true, "die", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(color, pattern, false, "bounding_box", gds_layer_idx, 1);
    } else if (gds_layer_idx == (gds_layer_size - 1)) {
      lyp_layer_list.emplace_back(color, pattern, false, "gcell", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(color, pattern, false, "gcell_text", gds_layer_idx, 1);
    } else {
      if (RTUtil::exist(_gds_routing_layer_map, gds_layer_idx)) {
        // routing
        std::string routing_layer_name = routing_layer_list[_gds_routing_layer_map[gds_layer_idx]].get_layer_name();
        for (auto& [routing_data_type, visible] : routing_data_type_visible_map) {
          lyp_layer_list.emplace_back(color, pattern, visible,
                                      RTUtil::getString(routing_layer_name, "_", GetGPLayoutTypeName()(routing_data_type)), gds_layer_idx,
                                      static_cast<irt_int>(routing_data_type));
        }
      } else if (RTUtil::exist(_gds_cut_layer_map, gds_layer_idx)) {
        // cut
        std::string cut_layer_name = cut_layer_list[_gds_cut_layer_map[gds_layer_idx]].get_layer_name();
        for (auto& [cut_data_type, visible] : cut_data_type_visible_map) {
          lyp_layer_list.emplace_back(color, pattern, visible, RTUtil::getString(cut_layer_name, "_", GetGPLayoutTypeName()(cut_data_type)),
                                      gds_layer_idx, static_cast<irt_int>(cut_data_type));
        }
      }
    }
  }
  writeLypFile(gp_temp_directory_path + "layout.lyp", lyp_layer_list);
}

void GDSPlotter::buildGraphLypFile()
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::string gp_temp_directory_path = DM_INST.getConfig().gp_temp_directory_path;

  std::vector<std::string> color_list = {"#ff9d9d", "#ff80a8", "#c080ff", "#9580ff", "#8086ff", "#80a8ff", "#ff0000", "#ff0080", "#ff00ff",
                                         "#8000ff", "#0000ff", "#0080ff", "#800000", "#800057", "#800080", "#500080", "#000080", "#004080",
                                         "#80fffb", "#80ff8d", "#afff80", "#f3ff80", "#ffc280", "#ffa080", "#00ffff", "#01ff6b", "#91ff00",
                                         "#ddff00", "#ffae00", "#ff8000", "#008080", "#008050", "#008000", "#508000", "#808000", "#805000"};
  std::vector<std::string> pattern_list = {"I5", "I9"};

  std::map<GPGraphType, bool> routing_data_type_visible_map
      = {{GPGraphType::kNone, false},       {GPGraphType::kOpen, false},      {GPGraphType::kClose, false},     {GPGraphType::kInfo, false},
         {GPGraphType::kNeighbor, false},   {GPGraphType::kKey, true},        {GPGraphType::kTrackAxis, false}, {GPGraphType::kPath, true},
         {GPGraphType::kLayoutShape, true}, {GPGraphType::kReservedVia, true}};
  std::map<GPGraphType, bool> cut_data_type_visible_map
      = {{GPGraphType::kPath, true}, {GPGraphType::kLayoutShape, true}, {GPGraphType::kReservedVia, true}};

  // 0为base_region 最后一个为GCell 中间为cut+routing
  irt_int gds_layer_size = 2 + static_cast<irt_int>(_gds_routing_layer_map.size() + _gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (irt_int gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    std::string color = color_list[gds_layer_idx % color_list.size()];
    std::string pattern = pattern_list[gds_layer_idx % pattern_list.size()];

    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(color, pattern, true, "base_region", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(color, pattern, false, "gcell", gds_layer_idx, 1);
      lyp_layer_list.emplace_back(color, pattern, false, "bounding_box", gds_layer_idx, 2);
    } else if (RTUtil::exist(_gds_routing_layer_map, gds_layer_idx)) {
      // routing
      std::string routing_layer_name = routing_layer_list[_gds_routing_layer_map[gds_layer_idx]].get_layer_name();
      for (auto& [routing_data_type, visible] : routing_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, pattern, visible,
                                    RTUtil::getString(routing_layer_name, "_", GetGPGraphTypeName()(routing_data_type)), gds_layer_idx,
                                    static_cast<irt_int>(routing_data_type));
      }
    } else if (RTUtil::exist(_gds_cut_layer_map, gds_layer_idx)) {
      // cut
      std::string cut_layer_name = cut_layer_list[_gds_cut_layer_map[gds_layer_idx]].get_layer_name();
      for (auto& [cut_data_type, visible] : cut_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, pattern, visible, RTUtil::getString(cut_layer_name, "_", GetGPGraphTypeName()(cut_data_type)),
                                    gds_layer_idx, static_cast<irt_int>(cut_data_type));
      }
    }
  }
  writeLypFile(gp_temp_directory_path + "graph.lyp", lyp_layer_list);
}

void GDSPlotter::writeLypFile(std::string lyp_file_path, std::vector<GPLYPLayer>& lyp_layer_list)
{
  std::ofstream* lyp_file = RTUtil::getOutputFileStream(lyp_file_path);
  RTUtil::pushStream(lyp_file, "<?xml version=\"1.0\" encoding=\"utf-8\"?>", "\n");
  RTUtil::pushStream(lyp_file, "<layer-properties>", "\n");

  for (size_t i = 0; i < lyp_layer_list.size(); i++) {
    GPLYPLayer& lyp_layer = lyp_layer_list[i];
    RTUtil::pushStream(lyp_file, "<properties>", "\n");
    RTUtil::pushStream(lyp_file, "<frame-color>", lyp_layer.get_color(), "</frame-color>", "\n");
    RTUtil::pushStream(lyp_file, "<fill-color>", lyp_layer.get_color(), "</fill-color>", "\n");
    RTUtil::pushStream(lyp_file, "<frame-brightness>0</frame-brightness>", "\n");
    RTUtil::pushStream(lyp_file, "<fill-brightness>0</fill-brightness>", "\n");
    RTUtil::pushStream(lyp_file, "<dither-pattern>", lyp_layer.get_pattern(), "</dither-pattern>", "\n");
    RTUtil::pushStream(lyp_file, "<line-style/>", "\n");
    RTUtil::pushStream(lyp_file, "<valid>true</valid>", "\n");
    if (lyp_layer.get_visible()) {
      RTUtil::pushStream(lyp_file, "<visible>true</visible>", "\n");
    } else {
      RTUtil::pushStream(lyp_file, "<visible>false</visible>", "\n");
    }
    RTUtil::pushStream(lyp_file, "<transparent>false</transparent>", "\n");
    RTUtil::pushStream(lyp_file, "<width/>", "\n");
    RTUtil::pushStream(lyp_file, "<marked>false</marked>", "\n");
    RTUtil::pushStream(lyp_file, "<xfill>false</xfill>", "\n");
    RTUtil::pushStream(lyp_file, "<animation>0</animation>", "\n");
    RTUtil::pushStream(lyp_file, "<name>", lyp_layer.get_layer_name(), " ", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(),
                       "</name>", "\n");
    RTUtil::pushStream(lyp_file, "<source>", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "@1</source>", "\n");
    RTUtil::pushStream(lyp_file, "</properties>", "\n");
  }
  RTUtil::pushStream(lyp_file, "</layer-properties>", "\n");
  RTUtil::closeFileStream(lyp_file);
}

void GDSPlotter::addNetList(GPGDS& gp_gds, std::vector<Net>& net_list, Stage stage)
{
  Monitor monitor;

  GPStruct net_list_struct(RTUtil::getString("net_list(size:", net_list.size(), ")"));
  for (size_t i = 0; i < net_list.size(); i++) {
    Net& net = net_list[i];

    GPStruct net_struct(
        RTUtil::getString("net_", net.get_net_idx(), "(", net.get_net_name(), ")", "(", GetConnectTypeName()(net.get_connect_type()), ")"),
        RTUtil::getString(net.get_net_idx()));

    addPinList(gp_gds, net_struct, net.get_pin_list());
    addBoundingBox(gp_gds, net_struct, net.get_bounding_box());

    switch (stage) {
      case Stage::kDetailedRouter:
        addRTNodeTree(gp_gds, net_struct, net.get_dr_result_tree());
        break;
      case Stage::kGlobalRouter:
        addRTNodeTree(gp_gds, net_struct, net.get_gr_result_tree());
        break;
      case Stage::kPinAccessor:
        break;
      case Stage::kTrackAssigner:
        addRTNodeTree(gp_gds, net_struct, net.get_ta_result_tree());
        break;
      case Stage::kViolationRepairer:
        addPHYNodeTree(gp_gds, net_struct, net.get_vr_result_tree());
        break;
      default:
        LOG_INST.error(Loc::current(), "Unknown stage type!");
        break;
    }
    net_list_struct.push(net_struct.get_name());
    gp_gds.addStruct(net_struct);
  }
  gp_gds.addStruct(net_list_struct);

  LOG_INST.info(Loc::current(), "Add ", net_list.size(), " nets to gds completed!", monitor.getStatsInfo());
}

void GDSPlotter::addPinList(GPGDS& gp_gds, GPStruct& net_struct, std::vector<Pin>& pin_list)
{
  GPStruct pin_list_struct(RTUtil::getString("pin_list@", net_struct.get_alias_name()));

  for (size_t i = 0; i < pin_list.size(); i++) {
    Pin& pin = pin_list[i];

    GPStruct pin_struct(RTUtil::getString("pin_", pin.get_pin_idx(), "(", pin.get_pin_name(), ")@", net_struct.get_alias_name()));

    addPinShapeList(pin_struct, pin);
    addAccessPointList(pin_struct, pin);
    pin_list_struct.push(pin_struct.get_name());
    gp_gds.addStruct(pin_struct);
  }
  net_struct.push(pin_list_struct.get_name());
  gp_gds.addStruct(pin_list_struct);
}

void GDSPlotter::addPinShapeList(GPStruct& pin_struct, Pin& pin)
{
  for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
    GPBoundary shape_boundary;
    shape_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_shape.get_layer_idx()));
    shape_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kPinShape));
    shape_boundary.set_rect(routing_shape.get_real_rect());
    pin_struct.push(shape_boundary);
  }
  for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
    GPBoundary shape_boundary;
    shape_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(cut_shape.get_layer_idx()));
    shape_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kPinShape));
    shape_boundary.set_rect(cut_shape.get_real_rect());
    pin_struct.push(shape_boundary);
  }
}

void GDSPlotter::addAccessPointList(GPStruct& pin_struct, Pin& pin)
{
  for (AccessPoint& access_point : pin.get_access_point_list()) {
    irt_int x = access_point.get_real_x();
    irt_int y = access_point.get_real_y();

    GPBoundary access_point_boundary;
    access_point_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(access_point.get_layer_idx()));
    access_point_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kAccessPoint));
    access_point_boundary.set_rect(x - 10, y - 10, x + 10, y + 10);
    pin_struct.push(access_point_boundary);
  }
}

void GDSPlotter::addBoundingBox(GPGDS& gp_gds, GPStruct& net_struct, BoundingBox& bounding_box)
{
  GPStruct bounding_box_struct(RTUtil::getString("bounding_box@", net_struct.get_alias_name()));

  GPBoundary bounding_box_boundary;
  bounding_box_boundary.set_layer_idx(0);
  bounding_box_boundary.set_data_type(1);
  bounding_box_boundary.set_rect(bounding_box.get_real_rect());
  bounding_box_struct.push(bounding_box_boundary);

  net_struct.push(bounding_box_struct.get_name());
  gp_gds.addStruct(bounding_box_struct);
}

void GDSPlotter::addRTNodeTree(GPGDS& gp_gds, GPStruct& net_struct, MTree<RTNode>& node_tree)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  GPStruct guide_list_struct(RTUtil::getString("guide_list@", net_struct.get_alias_name()));
  GPStruct routing_segment_list_struct(RTUtil::getString("routing_segment_list@", net_struct.get_alias_name()));

  for (TNode<RTNode>* rt_node_node : RTUtil::getNodeList(node_tree)) {
    RTNode& rt_node = rt_node_node->value();

    // guide_list
    Guide& first_guide = rt_node.get_first_guide();
    Guide& second_guide = rt_node.get_second_guide();
    irt_int first_layer_idx = first_guide.get_layer_idx();
    irt_int second_layer_idx = second_guide.get_layer_idx();

    PlanarRect real_rect = RTUtil::getRealRect(first_guide.get_grid_coord(), second_guide.get_grid_coord(), gcell_axis);

    if (first_layer_idx == second_layer_idx) {
      GPBoundary guide_boundary;
      guide_boundary.set_rect(real_rect);
      guide_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
      guide_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kGuide));
      guide_list_struct.push(guide_boundary);
    } else {
      RTUtil::swapASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        GPBoundary guide_boundary;
        guide_boundary.set_rect(real_rect);
        guide_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
        guide_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kGuide));
        guide_list_struct.push(guide_boundary);
      }
    }

    // routing_segment_list
    for (Segment<TNode<LayerCoord>*>& routing_segment : RTUtil::getSegListByTree(rt_node.get_routing_tree())) {
      LayerCoord& first_coord = routing_segment.get_first()->value();
      LayerCoord& second_coord = routing_segment.get_second()->value();

      irt_int first_layer_idx = first_coord.get_layer_idx();
      irt_int second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
        PlanarRect wire_rect = RTUtil::getEnlargedRect(first_coord, second_coord, half_width);

        GPBoundary wire_boundary;
        wire_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        wire_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kWire));
        wire_boundary.set_rect(wire_rect);
        routing_segment_list_struct.push(wire_boundary);
      } else {
        RTUtil::swapASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ViaMaster& via_master = layer_via_master_list[layer_idx].front();

          LayerRect& above_enclosure = via_master.get_above_enclosure();
          LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
          GPBoundary above_boundary;
          above_boundary.set_rect(offset_above_enclosure);
          above_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
          above_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kEnclosure));
          routing_segment_list_struct.push(above_boundary);

          LayerRect& below_enclosure = via_master.get_below_enclosure();
          LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
          GPBoundary below_boundary;
          below_boundary.set_rect(offset_below_enclosure);
          below_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
          below_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kEnclosure));
          routing_segment_list_struct.push(below_boundary);

          for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
            LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
            GPBoundary cut_boundary;
            cut_boundary.set_rect(offset_cut_shape);
            cut_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(via_master.get_cut_layer_idx()));
            cut_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kCut));
            routing_segment_list_struct.push(cut_boundary);
          }
        }
      }
    }
  }
  net_struct.push(guide_list_struct.get_name());
  net_struct.push(routing_segment_list_struct.get_name());
  gp_gds.addStruct(guide_list_struct);
  gp_gds.addStruct(routing_segment_list_struct);
}

void GDSPlotter::addPHYNodeTree(GPGDS& gp_gds, GPStruct& net_struct, MTree<PHYNode>& node_tree)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  GPStruct wire_list_struct(RTUtil::getString("wire_list@", net_struct.get_alias_name()));
  GPStruct via_list_struct(RTUtil::getString("via_list@", net_struct.get_alias_name()));
  GPStruct patch_list_struct(RTUtil::getString("patch_list@", net_struct.get_alias_name()));

  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(node_tree)) {
    PHYNode& phy_node = phy_node_node->value();

    if (phy_node.isType<WireNode>()) {
      WireNode& wire_node = phy_node.getNode<WireNode>();
      PlanarRect wire_rect = RTUtil::getEnlargedRect(wire_node.get_first(), wire_node.get_second(), wire_node.get_wire_width() / 2);

      GPBoundary wire_boundary;
      wire_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(wire_node.get_layer_idx()));
      wire_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kWire));
      wire_boundary.set_rect(wire_rect);
      wire_list_struct.push(wire_boundary);

    } else if (phy_node.isType<ViaNode>()) {
      ViaNode& via_node = phy_node.getNode<ViaNode>();
      ViaMasterIdx& via_master_idx = via_node.get_via_master_idx();
      ViaMaster& via_master = layer_via_master_list[via_master_idx.get_below_layer_idx()][via_master_idx.get_via_idx()];

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      GPBoundary above_boundary;
      above_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
      above_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kEnclosure));
      above_boundary.set_rect(RTUtil::getOffsetRect(above_enclosure, via_node));
      via_list_struct.push(above_boundary);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      GPBoundary below_boundary;
      below_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
      below_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kEnclosure));
      below_boundary.set_rect(RTUtil::getOffsetRect(below_enclosure, via_node));
      via_list_struct.push(below_boundary);

      std::vector<PlanarRect>& cut_shape_list = via_master.get_cut_shape_list();
      for (PlanarRect& cut_shape : cut_shape_list) {
        GPBoundary cut_boundary;
        cut_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(via_master.get_cut_layer_idx()));
        cut_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kCut));
        cut_boundary.set_rect(RTUtil::getOffsetRect(cut_shape, via_node));
        via_list_struct.push(cut_boundary);
      }
    } else if (phy_node.isType<PatchNode>()) {
      PatchNode& patch_node = phy_node.getNode<PatchNode>();

      GPBoundary patch_boundary;
      patch_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(patch_node.get_layer_idx()));
      patch_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kPatch));
      patch_boundary.set_rect(patch_node);
      patch_list_struct.push(patch_boundary);

    } else if (phy_node.isType<PinNode>()) {
      continue;
    } else {
      LOG_INST.error(Loc::current(), "Incorrect phy node type!");
    }
  }
  net_struct.push(wire_list_struct.get_name());
  net_struct.push(via_list_struct.get_name());
  net_struct.push(patch_list_struct.get_name());
  gp_gds.addStruct(wire_list_struct);
  gp_gds.addStruct(via_list_struct);
  gp_gds.addStruct(patch_list_struct);
}

void GDSPlotter::addCostMap(GPGDS& gp_gds, std::vector<Net>& net_list)
{
  Monitor monitor;

  GPStruct net_list_struct(RTUtil::getString("net_list(size:", net_list.size(), ")"));
  for (size_t i = 0; i < net_list.size(); i++) {
    Net& net = net_list[i];

    GPStruct net_struct(
        RTUtil::getString("net_", net.get_net_idx(), "(", net.get_net_name(), ")", "(", GetConnectTypeName()(net.get_connect_type()), ")"),
        RTUtil::getString(net.get_net_idx()));

    addBoundingBox(gp_gds, net_struct, net.get_bounding_box());
    addCostMap(gp_gds, net_struct, net.get_bounding_box(), net.get_ra_cost_map());

    net_list_struct.push(net_struct.get_name());
    gp_gds.addStruct(net_struct);
  }
  gp_gds.addStruct(net_list_struct);

  LOG_INST.info(Loc::current(), "Add ", net_list.size(), " nets to gds completed!", monitor.getStatsInfo());
}

void GDSPlotter::addCostMap(GPGDS& gp_gds, GPStruct& net_struct, BoundingBox& bounding_box, GridMap<double>& cost_map)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

  GPStruct cost_map_struct(RTUtil::getString("cost_map@", net_struct.get_alias_name()));

  for (irt_int x = 0; x < cost_map.get_x_size(); ++x) {
    for (irt_int y = 0; y < cost_map.get_y_size(); ++y) {
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(static_cast<irt_int>(cost_map[x][y] * 100));
      gp_boundary.set_data_type(0);
      gp_boundary.set_rect(
          RTUtil::getRealRect(PlanarCoord(x + bounding_box.get_grid_lb_x(), y + bounding_box.get_grid_lb_y()), gcell_axis));
      cost_map_struct.push(gp_boundary);
    }
  }
  net_struct.push(cost_map_struct.get_name());
  gp_gds.addStruct(cost_map_struct);
}

void GDSPlotter::plotGDS(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping)
{
  Die& die = DM_INST.getDatabase().get_die();

  if (add_layout) {
    PlanarRect clipping_window;
    if (need_clipping) {
      clipping_window = getClippingWindow(gp_gds);
    } else {
      clipping_window = die.get_real_rect();
    }
    addLayout(gp_gds, clipping_window);
  }
  buildAndCheckGDS(gp_gds);
  plotGDS(gds_file_path, gp_gds);
}

PlanarRect GDSPlotter::getClippingWindow(GPGDS& gp_gds)
{
  size_t coord_num = 0;
  for (GPStruct& gp_struct : gp_gds.get_struct_list()) {
    coord_num += (gp_struct.get_boundary_list().size() * 2);
    coord_num += (gp_struct.get_path_list().size() * 2);
    coord_num += gp_struct.get_text_list().size();
  }
  std::vector<PlanarCoord> coord_list;
  coord_list.reserve(coord_num);
  for (GPStruct& gp_struct : gp_gds.get_struct_list()) {
    for (GPBoundary& gp_boundary : gp_struct.get_boundary_list()) {
      coord_list.push_back(gp_boundary.get_lb());
      coord_list.push_back(gp_boundary.get_rt());
    }
    for (GPPath& gp_path : gp_struct.get_path_list()) {
      coord_list.push_back(gp_path.get_first());
      coord_list.push_back(gp_path.get_second());
    }
    for (GPText& gp_text : gp_struct.get_text_list()) {
      coord_list.push_back(gp_text.get_coord());
    }
  }
  return RTUtil::getBoundingBox(coord_list);
}

void GDSPlotter::addLayout(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  Monitor monitor;

  addDie(gp_gds, clipping_window);
  addGCellAxis(gp_gds, clipping_window);
  addTrackGrid(gp_gds, clipping_window);
  addBlockageList(gp_gds, clipping_window);

  LOG_INST.info(Loc::current(), "Add layout to gds completed!", monitor.getStatsInfo());
}

void GDSPlotter::addDie(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  if (!RTUtil::isOpenOverlap(die.get_real_rect(), clipping_window)) {
    return;
  }

  GPStruct die_struct("die");

  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(RTUtil::getOverlap(die.get_real_rect(), clipping_window));
  die_struct.push(gp_boundary);
  gp_gds.addStruct(die_struct);
}

void GDSPlotter::addGCellAxis(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  irt_int gcell_axis_layer_idx = static_cast<irt_int>(routing_layer_list.size() + cut_layer_list.size() + 1);
  irt_int window_lb_x = clipping_window.get_lb_x();
  irt_int window_lb_y = clipping_window.get_lb_y();
  irt_int window_rt_x = clipping_window.get_rt_x();
  irt_int window_rt_y = clipping_window.get_rt_y();

  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    ScaleGrid& gcell_grid = x_grid_list[i];
    for (irt_int x = gcell_grid.get_start_line(); x <= gcell_grid.get_end_line(); x += gcell_grid.get_step_length()) {
      if (x < window_lb_x || window_rt_x < x) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(gcell_axis_layer_idx);
      gp_path.set_data_type(static_cast<irt_int>(GPLayoutType::kNone));
      gp_path.set_segment(x, window_lb_y, x, window_rt_y);
      gcell_axis_struct.push(gp_path);
    }
  }
  std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();
  for (size_t i = 0; i < y_grid_list.size(); i++) {
    ScaleGrid& gcell_grid = y_grid_list[i];
    for (irt_int y = gcell_grid.get_start_line(); y <= gcell_grid.get_end_line(); y += gcell_grid.get_step_length()) {
      if (y < window_lb_y || window_rt_y < y) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(gcell_axis_layer_idx);
      gp_path.set_data_type(static_cast<irt_int>(GPLayoutType::kNone));
      gp_path.set_segment(window_lb_x, y, window_rt_x, y);
      gcell_axis_struct.push(gp_path);
    }
  }
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    ScaleGrid& x_gcell_grid = x_grid_list[i];
    for (size_t j = 0; j < y_grid_list.size(); j++) {
      ScaleGrid& y_gcell_grid = y_grid_list[j];
      for (irt_int x = x_gcell_grid.get_start_line(); x < x_gcell_grid.get_end_line(); x += x_gcell_grid.get_step_length()) {
        for (irt_int y = y_gcell_grid.get_start_line(); y < y_gcell_grid.get_end_line(); y += y_gcell_grid.get_step_length()) {
          if (x < window_lb_x || window_rt_x < x || y < window_lb_y || window_rt_y < y) {
            continue;
          }
          GPText gp_text;
          gp_text.set_layer_idx(gcell_axis_layer_idx);
          gp_text.set_text_type(static_cast<irt_int>(GPLayoutType::kText));
          gp_text.set_presentation(GPTextPresentation::kLeftBottom);
          gp_text.set_coord(x, y);
          gp_text.set_message(RTUtil::getString("grid(", RTUtil::getGridLB(x, x_grid_list), " , ", RTUtil::getGridLB(y, y_grid_list),
                                                ") real(", x, " , ", y, ")"));
          gcell_axis_struct.push(gp_text);
        }
      }
    }
  }
  gp_gds.addStruct(gcell_axis_struct);
}

void GDSPlotter::addTrackGrid(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  irt_int window_lb_x = clipping_window.get_lb_x();
  irt_int window_lb_y = clipping_window.get_lb_y();
  irt_int window_rt_x = clipping_window.get_rt_x();
  irt_int window_rt_y = clipping_window.get_rt_y();

  GPStruct track_grid_struct("track_grid");

  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    RoutingLayer& routing_layer = routing_layer_list[i];
    irt_int layer_idx = routing_layer.get_layer_idx();

    ScaleGrid& x_track_grid = routing_layer.getXTrackGrid();
    ScaleGrid& y_track_grid = routing_layer.getYTrackGrid();

    GPLayoutType x_data_type = GPLayoutType::kPreferTrack;
    GPLayoutType y_data_type = GPLayoutType::kNonpreferTrack;
    if (routing_layer.isPreferH()) {
      std::swap(x_data_type, y_data_type);
    }
    for (irt_int x = x_track_grid.get_start_line(); x <= x_track_grid.get_end_line(); x += x_track_grid.get_step_length()) {
      if (x < window_lb_x || window_rt_x < x) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      gp_path.set_data_type(static_cast<irt_int>(x_data_type));
      gp_path.set_segment(x, window_lb_y, x, window_rt_y);
      track_grid_struct.push(gp_path);
    }
    for (irt_int y = y_track_grid.get_start_line(); y <= y_track_grid.get_end_line(); y += y_track_grid.get_step_length()) {
      if (y < window_lb_y || window_rt_y < y) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      gp_path.set_data_type(static_cast<irt_int>(y_data_type));
      gp_path.set_segment(window_lb_x, y, window_rt_x, y);
      track_grid_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(track_grid_struct);
}

void GDSPlotter::addBlockageList(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = DM_INST.getDatabase().get_cut_blockage_list();

  GPStruct layout_blockage_struct("layout_blockage");

  for (Blockage& blockage : routing_blockage_list) {
    if (!RTUtil::isOpenOverlap(blockage.get_real_rect(), clipping_window)) {
      continue;
    }
    GPBoundary blockage_boundary;
    blockage_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(blockage.get_layer_idx()));
    blockage_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kBlockage));
    blockage_boundary.set_rect(RTUtil::getOverlap(blockage.get_real_rect(), clipping_window));
    layout_blockage_struct.push(blockage_boundary);
  }
  for (Blockage& blockage : cut_blockage_list) {
    if (!RTUtil::isOpenOverlap(blockage.get_real_rect(), clipping_window)) {
      continue;
    }
    GPBoundary blockage_boundary;
    blockage_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(blockage.get_layer_idx()));
    blockage_boundary.set_data_type(static_cast<irt_int>(GPLayoutType::kBlockage));
    blockage_boundary.set_rect(RTUtil::getOverlap(blockage.get_real_rect(), clipping_window));
    layout_blockage_struct.push(blockage_boundary);
  }
  gp_gds.addStruct(layout_blockage_struct);

  GPStruct blockage_list_struct("blockage_list");
  blockage_list_struct.push(layout_blockage_struct.get_name());
  gp_gds.addStruct(blockage_list_struct);
}

void GDSPlotter::buildAndCheckGDS(GPGDS& gp_gds)
{
  buildTopStruct(gp_gds);
  checkSRefList(gp_gds);
}

void GDSPlotter::buildTopStruct(GPGDS& gp_gds)
{
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();

  std::set<std::string> unrefed_struct_name_set;
  for (GPStruct& gp_struct : struct_list) {
    unrefed_struct_name_set.insert(gp_struct.get_name());
  }

  for (GPStruct& gp_struct : struct_list) {
    std::vector<std::string>& sref_name_list = gp_struct.get_sref_name_list();
    for (std::string& sref_name : sref_name_list) {
      unrefed_struct_name_set.erase(sref_name);
    }
  }

  GPStruct top_struct(gp_gds.get_top_name());
  for (const std::string& unrefed_struct_name : unrefed_struct_name_set) {
    top_struct.push(unrefed_struct_name);
  }
  gp_gds.addStruct(top_struct);
}

void GDSPlotter::checkSRefList(GPGDS& gp_gds)
{
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();

  std::set<std::string> nonexistent_sref_name_set;
  for (GPStruct& gp_struct : struct_list) {
    for (std::string& sref_name : gp_struct.get_sref_name_list()) {
      nonexistent_sref_name_set.insert(sref_name);
    }
  }
  for (GPStruct& gp_struct : struct_list) {
    nonexistent_sref_name_set.erase(gp_struct.get_name());
  }

  if (!nonexistent_sref_name_set.empty()) {
    for (const std::string& nonexistent_sref_name : nonexistent_sref_name_set) {
      LOG_INST.warning(Loc::current(), "There is no corresponding structure ", nonexistent_sref_name, " in GDS!");
    }
    LOG_INST.error(Loc::current(), "There is a non-existent structure reference!");
  }
}

void GDSPlotter::plotGDS(std::string gds_file_path, GPGDS& gp_gds)
{
  Monitor monitor;

  LOG_INST.info(Loc::current(), "The gds file is being saved...");

  std::ofstream* gds_file = RTUtil::getOutputFileStream(gds_file_path);
  RTUtil::pushStream(gds_file, "HEADER 600", "\n");
  RTUtil::pushStream(gds_file, "BGNLIB", "\n");
  RTUtil::pushStream(gds_file, "LIBNAME ", gp_gds.get_top_name(), "\n");
  RTUtil::pushStream(gds_file, "UNITS 0.001 1e-9", "\n");
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();
  for (size_t i = 0; i < struct_list.size(); i++) {
    plotStruct(gds_file, struct_list[i]);
  }
  RTUtil::pushStream(gds_file, "ENDLIB", "\n");
  RTUtil::closeFileStream(gds_file);

  LOG_INST.info(Loc::current(), "The gds file has been saved in '", gds_file_path, "'!", monitor.getStatsInfo());
}

void GDSPlotter::plotStruct(std::ofstream* gds_file, GPStruct& gp_struct)
{
  RTUtil::pushStream(gds_file, "BGNSTR", "\n");
  RTUtil::pushStream(gds_file, "STRNAME ", gp_struct.get_name(), "\n");
  // boundary
  for (GPBoundary& gp_boundary : gp_struct.get_boundary_list()) {
    plotBoundary(gds_file, gp_boundary);
  }
  // path
  for (GPPath& gp_path : gp_struct.get_path_list()) {
    plotPath(gds_file, gp_path);
  }
  // text
  for (GPText& gp_text : gp_struct.get_text_list()) {
    plotText(gds_file, gp_text);
  }
  // sref
  for (std::string& sref_name : gp_struct.get_sref_name_list()) {
    plotSref(gds_file, sref_name);
  }
  RTUtil::pushStream(gds_file, "ENDSTR", "\n");
}

void GDSPlotter::plotBoundary(std::ofstream* gds_file, GPBoundary& gp_boundary)
{
  irt_int lb_x = gp_boundary.get_lb_x();
  irt_int lb_y = gp_boundary.get_lb_y();
  irt_int rt_x = gp_boundary.get_rt_x();
  irt_int rt_y = gp_boundary.get_rt_y();

  RTUtil::pushStream(gds_file, "BOUNDARY", "\n");
  RTUtil::pushStream(gds_file, "LAYER ", gp_boundary.get_layer_idx(), "\n");
  RTUtil::pushStream(gds_file, "DATATYPE ", static_cast<irt_int>(gp_boundary.get_data_type()), "\n");
  RTUtil::pushStream(gds_file, "XY", "\n");
  RTUtil::pushStream(gds_file, lb_x, " : ", lb_y, "\n");
  RTUtil::pushStream(gds_file, rt_x, " : ", lb_y, "\n");
  RTUtil::pushStream(gds_file, rt_x, " : ", rt_y, "\n");
  RTUtil::pushStream(gds_file, lb_x, " : ", rt_y, "\n");
  RTUtil::pushStream(gds_file, lb_x, " : ", lb_y, "\n");
  RTUtil::pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotPath(std::ofstream* gds_file, GPPath& gp_path)
{
  Segment<PlanarCoord>& segment = gp_path.get_segment();
  irt_int first_x = segment.get_first().get_x();
  irt_int first_y = segment.get_first().get_y();
  irt_int second_x = segment.get_second().get_x();
  irt_int second_y = segment.get_second().get_y();

  RTUtil::pushStream(gds_file, "PATH", "\n");
  RTUtil::pushStream(gds_file, "LAYER ", gp_path.get_layer_idx(), "\n");
  RTUtil::pushStream(gds_file, "DATATYPE ", static_cast<irt_int>(gp_path.get_data_type()), "\n");
  RTUtil::pushStream(gds_file, "WIDTH ", gp_path.get_width(), "\n");
  RTUtil::pushStream(gds_file, "XY", "\n");
  RTUtil::pushStream(gds_file, first_x, " : ", first_y, "\n");
  RTUtil::pushStream(gds_file, second_x, " : ", second_y, "\n");
  RTUtil::pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotText(std::ofstream* gds_file, GPText& gp_text)
{
  PlanarCoord& coord = gp_text.get_coord();
  irt_int x = coord.get_x();
  irt_int y = coord.get_y();

  RTUtil::pushStream(gds_file, "TEXT", "\n");
  RTUtil::pushStream(gds_file, "LAYER ", gp_text.get_layer_idx(), "\n");
  RTUtil::pushStream(gds_file, "TEXTTYPE ", gp_text.get_text_type(), "\n");
  RTUtil::pushStream(gds_file, "PRESENTATION ", static_cast<irt_int>(gp_text.get_presentation()), "\n");
  RTUtil::pushStream(gds_file, "XY", "\n");
  RTUtil::pushStream(gds_file, x, " : ", y, "\n");
  RTUtil::pushStream(gds_file, "STRING ", gp_text.get_message(), "\n");
  RTUtil::pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotSref(std::ofstream* gds_file, std::string& sref_name)
{
  RTUtil::pushStream(gds_file, "SREF", "\n");
  RTUtil::pushStream(gds_file, "SNAME ", sref_name, "\n");
  RTUtil::pushStream(gds_file, "XY 0:0", "\n");
  RTUtil::pushStream(gds_file, "ENDEL", "\n");
}

}  // namespace irt
