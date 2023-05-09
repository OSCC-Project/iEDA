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

#include "MTree.hpp"
#include "Monitor.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void GDSPlotter::initInst(Config& config, Database& database)
{
  if (_gp_instance == nullptr) {
    _gp_instance = new GDSPlotter(config, database);
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

void GDSPlotter::plot(Net& net, Stage stage, bool add_layout, bool need_clipping)
{
  std::vector<Net> net_list = {net};
  plot(net_list, stage, add_layout, need_clipping);
}

void GDSPlotter::plot(std::vector<Net>& net_list, Stage stage, bool add_layout, bool need_clipping)
{
  std::string gds_file_path = _gp_data_manager.getConfig().temp_directory_path + GetStageName()(stage) + ".gds";

  GPGDS gp_gds;
  addNetList(gp_gds, net_list, stage);
  plot(gp_gds, gds_file_path, add_layout, need_clipping);
}

void GDSPlotter::plot(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping)
{
  plotGDS(gp_gds, gds_file_path, add_layout, need_clipping);
}

irt_int GDSPlotter::getGDSIdxByRouting(irt_int routing_layer_idx)
{
  irt_int gds_layer_idx = 0;
  std::map<irt_int, irt_int>& routing_layer_gds_map = _gp_data_manager.getDatabase().get_routing_layer_gds_map();
  if (RTUtil::exist(routing_layer_gds_map, routing_layer_idx)) {
    gds_layer_idx = routing_layer_gds_map[routing_layer_idx];
  } else {
    LOG_INST.warning(Loc::current(), "The routing_layer_idx '", routing_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

irt_int GDSPlotter::getGDSIdxByCut(irt_int cut_layer_idx)
{
  irt_int gds_layer_idx = 0;
  std::map<irt_int, irt_int>& cut_layer_gds_map = _gp_data_manager.getDatabase().get_cut_layer_gds_map();
  if (RTUtil::exist(cut_layer_gds_map, cut_layer_idx)) {
    gds_layer_idx = cut_layer_gds_map[cut_layer_idx];
  } else {
    LOG_INST.warning(Loc::current(), "The cut_layer_idx '", cut_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

// private

GDSPlotter* GDSPlotter::_gp_instance = nullptr;

void GDSPlotter::init(Config& config, Database& database)
{
  _gp_data_manager.input(config, database);
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
        // LOG_INST.error(Loc::current(), "Unknown process stage type!");
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
    shape_boundary.set_data_type(static_cast<irt_int>(GPDataType::kPort));
    shape_boundary.set_rect(routing_shape.get_real_rect());
    pin_struct.push(shape_boundary);
  }
  for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
    GPBoundary shape_boundary;
    shape_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(cut_shape.get_layer_idx()));
    shape_boundary.set_data_type(static_cast<irt_int>(GPDataType::kPort));
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
    access_point_boundary.set_data_type(static_cast<irt_int>(GPDataType::kAccessPoint));
    access_point_boundary.set_rect(x - 10, y - 10, x + 10, y + 10);
    pin_struct.push(access_point_boundary);
  }
}

void GDSPlotter::addBoundingBox(GPGDS& gp_gds, GPStruct& net_struct, BoundingBox& bounding_box)
{
  std::vector<RoutingLayer>& routing_layer_list = _gp_data_manager.getDatabase().get_routing_layer_list();

  GPStruct bounding_box_struct(RTUtil::getString("bounding_box@", net_struct.get_alias_name()));

  GPBoundary bounding_box_boundary;
  bounding_box_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer_list.front().get_layer_idx()));
  bounding_box_boundary.set_data_type(static_cast<irt_int>(GPDataType::kBoundingBox));
  bounding_box_boundary.set_rect(bounding_box.get_real_rect());
  bounding_box_struct.push(bounding_box_boundary);

  net_struct.push(bounding_box_struct.get_name());
  gp_gds.addStruct(bounding_box_struct);
}

void GDSPlotter::addRTNodeTree(GPGDS& gp_gds, GPStruct& net_struct, MTree<RTNode>& node_tree)
{
  GCellAxis& gcell_axis = _gp_data_manager.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = _gp_data_manager.getDatabase().get_routing_layer_list();

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
      guide_boundary.set_data_type(static_cast<irt_int>(GPDataType::kGuide));
      guide_list_struct.push(guide_boundary);
    } else {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        GPBoundary guide_boundary;
        guide_boundary.set_rect(real_rect);
        guide_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
        guide_boundary.set_data_type(static_cast<irt_int>(GPDataType::kGuide));
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
        GPPath wire_path;
        wire_path.set_segment(first_coord, second_coord);
        wire_path.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        wire_path.set_data_type(static_cast<irt_int>(GPDataType::kWire));
        wire_path.set_width(routing_layer_list[first_layer_idx].get_min_width());
        routing_segment_list_struct.push(wire_path);
      } else {
        RTUtil::sortASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          GPBoundary via_boundary;
          via_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, routing_layer_list[layer_idx].get_min_width() / 2));
          via_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(layer_idx));
          via_boundary.set_data_type(static_cast<irt_int>(GPDataType::kCut));
          routing_segment_list_struct.push(via_boundary);
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
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _gp_data_manager.getDatabase().get_layer_via_master_list();

  GPStruct wire_list_struct(RTUtil::getString("wire_list@", net_struct.get_alias_name()));
  GPStruct via_list_struct(RTUtil::getString("via_list@", net_struct.get_alias_name()));

  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(node_tree)) {
    PHYNode& phy_node = phy_node_node->value();

    if (phy_node.isType<WireNode>()) {
      WireNode& wire_node = phy_node.getNode<WireNode>();

      GPPath wire_path;
      wire_path.set_layer_idx(GP_INST.getGDSIdxByRouting(wire_node.get_layer_idx()));
      wire_path.set_data_type(static_cast<irt_int>(GPDataType::kWire));
      wire_path.set_width(wire_node.get_wire_width());
      wire_path.set_segment(wire_node.get_first().get_real_coord(), wire_node.get_second().get_real_coord());
      wire_list_struct.push(wire_path);

    } else if (phy_node.isType<ViaNode>()) {
      ViaNode& via_node = phy_node.getNode<ViaNode>();
      ViaMaster& via_master = layer_via_master_list[via_node.get_via_idx().first][via_node.get_via_idx().second];

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      GPBoundary above_boundary;
      above_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
      above_boundary.set_data_type(static_cast<irt_int>(GPDataType::kEnclosure));
      above_boundary.set_rect(RTUtil::getOffsetRect(above_enclosure, via_node.get_real_coord()));
      via_list_struct.push(above_boundary);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      GPBoundary below_boundary;
      below_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
      below_boundary.set_data_type(static_cast<irt_int>(GPDataType::kEnclosure));
      below_boundary.set_rect(RTUtil::getOffsetRect(below_enclosure, via_node.get_real_coord()));
      via_list_struct.push(below_boundary);

      std::vector<PlanarRect>& cut_shape_list = via_master.get_cut_shape_list();
      for (PlanarRect& cut_shape : cut_shape_list) {
        GPBoundary cut_boundary;
        cut_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(via_master.get_cut_layer_idx()));
        cut_boundary.set_data_type(static_cast<irt_int>(GPDataType::kCut));
        cut_boundary.set_rect(RTUtil::getOffsetRect(cut_shape, via_node.get_real_coord()));
        via_list_struct.push(cut_boundary);
      }
    } else if (phy_node.isType<PinNode>()) {
      continue;
    } else {
      LOG_INST.error(Loc::current(), "Incorrect phy node type!");
    }
  }
  net_struct.push(wire_list_struct.get_name());
  net_struct.push(via_list_struct.get_name());
  gp_gds.addStruct(wire_list_struct);
  gp_gds.addStruct(via_list_struct);
}

void GDSPlotter::plotGDS(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping)
{
  if (add_layout) {
    PlanarRect clipping_window;
    if (need_clipping) {
      clipping_window = getClippingWindow(gp_gds);
    } else {
      clipping_window = _gp_data_manager.getDatabase().get_die().get_real_rect();
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
  EXTPlanarRect& die = _gp_data_manager.getDatabase().get_die();

  if (!RTUtil::isOpenOverlap(die.get_real_rect(), clipping_window)) {
    return;
  }

  GPStruct die_struct("die");

  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kNone));
  gp_boundary.set_rect(RTUtil::getOverlap(die.get_real_rect(), clipping_window));
  die_struct.push(gp_boundary);
  gp_gds.addStruct(die_struct);
}

void GDSPlotter::addGCellAxis(GPGDS& gp_gds, PlanarRect& clipping_window)
{
  GPDatabase& database = _gp_data_manager.getDatabase();
  GCellAxis& gcell_axis = database.get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = database.get_cut_layer_list();

  irt_int gcell_axis_layer_idx = static_cast<irt_int>(routing_layer_list.size() + cut_layer_list.size() + 1);
  irt_int window_lb_x = clipping_window.get_lb_x();
  irt_int window_lb_y = clipping_window.get_lb_y();
  irt_int window_rt_x = clipping_window.get_rt_x();
  irt_int window_rt_y = clipping_window.get_rt_y();

  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<GCellGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    GCellGrid& gcell_grid = x_grid_list[i];
    for (irt_int x = gcell_grid.get_start_line(); x <= gcell_grid.get_end_line(); x += gcell_grid.get_step_length()) {
      if (x < window_lb_x || window_rt_x < x) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(gcell_axis_layer_idx);
      gp_path.set_data_type(static_cast<irt_int>(GPDataType::kNone));
      gp_path.set_segment(x, window_lb_y, x, window_rt_y);
      gcell_axis_struct.push(gp_path);
    }
  }
  std::vector<GCellGrid>& y_grid_list = gcell_axis.get_y_grid_list();
  for (size_t i = 0; i < y_grid_list.size(); i++) {
    GCellGrid& gcell_grid = y_grid_list[i];
    for (irt_int y = gcell_grid.get_start_line(); y <= gcell_grid.get_end_line(); y += gcell_grid.get_step_length()) {
      if (y < window_lb_y || window_rt_y < y) {
        continue;
      }
      GPPath gp_path;
      gp_path.set_layer_idx(gcell_axis_layer_idx);
      gp_path.set_data_type(static_cast<irt_int>(GPDataType::kNone));
      gp_path.set_segment(window_lb_x, y, window_rt_x, y);
      gcell_axis_struct.push(gp_path);
    }
  }
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    GCellGrid& x_gcell_grid = x_grid_list[i];
    for (size_t j = 0; j < y_grid_list.size(); j++) {
      GCellGrid& y_gcell_grid = y_grid_list[j];
      for (irt_int x = x_gcell_grid.get_start_line(); x < x_gcell_grid.get_end_line(); x += x_gcell_grid.get_step_length()) {
        for (irt_int y = y_gcell_grid.get_start_line(); y < y_gcell_grid.get_end_line(); y += y_gcell_grid.get_step_length()) {
          if (x < window_lb_x || window_rt_x < x || y < window_lb_y || window_rt_y < y) {
            continue;
          }
          GPText gp_text;
          gp_text.set_layer_idx(gcell_axis_layer_idx);
          gp_text.set_text_type(static_cast<irt_int>(GPDataType::kText));
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
  GPDatabase& database = _gp_data_manager.getDatabase();
  std::vector<RoutingLayer>& routing_layer_list = database.get_routing_layer_list();
  irt_int window_lb_x = clipping_window.get_lb_x();
  irt_int window_lb_y = clipping_window.get_lb_y();
  irt_int window_rt_x = clipping_window.get_rt_x();
  irt_int window_rt_y = clipping_window.get_rt_y();

  GPStruct track_grid_struct("track_grid");

  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    RoutingLayer& routing_layer = routing_layer_list[i];
    irt_int layer_idx = routing_layer.get_layer_idx();

    TrackGrid& x_track_grid = routing_layer.getXTrackGrid();
    TrackGrid& y_track_grid = routing_layer.getYTrackGrid();

    GPDataType x_data_type = GPDataType::kPreferTrack;
    GPDataType y_data_type = GPDataType::kNonpreferTrack;
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
  GPDatabase& database = _gp_data_manager.getDatabase();

  GPStruct artificial_blockage_struct("artificial_blockage");
  GPStruct layout_blockage_struct("layout_blockage");

  for (Blockage& blockage : database.get_routing_blockage_list()) {
    if (!RTUtil::isOpenOverlap(blockage.get_real_rect(), clipping_window)) {
      continue;
    }
    GPBoundary blockage_boundary;
    blockage_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(blockage.get_layer_idx()));
    blockage_boundary.set_data_type(static_cast<irt_int>(GPDataType::kBlockage));
    blockage_boundary.set_rect(RTUtil::getOverlap(blockage.get_real_rect(), clipping_window));

    if (blockage.isArtificial()) {
      artificial_blockage_struct.push(blockage_boundary);
    } else {
      layout_blockage_struct.push(blockage_boundary);
    }
  }
  for (Blockage& blockage : database.get_cut_blockage_list()) {
    if (!RTUtil::isOpenOverlap(blockage.get_real_rect(), clipping_window)) {
      continue;
    }
    GPBoundary blockage_boundary;
    blockage_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(blockage.get_layer_idx()));
    blockage_boundary.set_data_type(static_cast<irt_int>(GPDataType::kBlockage));
    blockage_boundary.set_rect(RTUtil::getOverlap(blockage.get_real_rect(), clipping_window));

    if (blockage.isArtificial()) {
      artificial_blockage_struct.push(blockage_boundary);
    } else {
      layout_blockage_struct.push(blockage_boundary);
    }
  }
  gp_gds.addStruct(artificial_blockage_struct);
  gp_gds.addStruct(layout_blockage_struct);

  GPStruct blockage_list_struct("blockage_list");
  blockage_list_struct.push(artificial_blockage_struct.get_name());
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
