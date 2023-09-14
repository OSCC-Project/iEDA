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
#include "ViolationRepairer.hpp"

namespace irt {

// public

void ViolationRepairer::initInst()
{
  if (_vr_instance == nullptr) {
    _vr_instance = new ViolationRepairer();
  }
}

ViolationRepairer& ViolationRepairer::getInst()
{
  if (_vr_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_vr_instance;
}

void ViolationRepairer::destroyInst()
{
  if (_vr_instance != nullptr) {
    delete _vr_instance;
    _vr_instance = nullptr;
  }
}

// function

void ViolationRepairer::repair(std::vector<Net>& net_list)
{
  Monitor monitor;

  repairNetList(net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kViolationRepairer), " completed!", monitor.getStatsInfo());
}

// private

ViolationRepairer* ViolationRepairer::_vr_instance = nullptr;

void ViolationRepairer::repairNetList(std::vector<Net>& net_list)
{
  VRModel vr_model = init(net_list);
  iterative(vr_model);
  update(vr_model);
}

#if 1  // init

VRModel ViolationRepairer::init(std::vector<Net>& net_list)
{
  VRModel vr_model = initVRModel(net_list);
  buildVRModel(vr_model);
  checkVRModel(vr_model);
  return vr_model;
}

VRModel ViolationRepairer::initVRModel(std::vector<Net>& net_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  VRModel vr_model;
  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();
  vr_gcell_map.init(die.getXSize(), die.getYSize());
  for (irt_int x = 0; x < die.getXSize(); x++) {
    for (irt_int y = 0; y < die.getYSize(); y++) {
      VRGCell& vr_gcell = vr_gcell_map[x][y];
      vr_gcell.set_base_region(RTUtil::getRealRect(x, y, gcell_axis));
      vr_gcell.set_top_layer_idx(routing_layer_list.back().get_layer_idx());
      vr_gcell.set_bottom_layer_idx(routing_layer_list.front().get_layer_idx());
    }
  }
  vr_model.set_vr_net_list(convertToVRNetList(net_list));

  return vr_model;
}

std::vector<VRNet> ViolationRepairer::convertToVRNetList(std::vector<Net>& net_list)
{
  std::vector<VRNet> vr_net_list;
  vr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    vr_net_list.emplace_back(convertToVRNet(net));
  }
  return vr_net_list;
}

VRNet ViolationRepairer::convertToVRNet(Net& net)
{
  VRNet vr_net;
  vr_net.set_origin_net(&net);
  vr_net.set_net_idx(net.get_net_idx());
  vr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    vr_net.get_vr_pin_list().push_back(VRPin(pin));
  }
  vr_net.set_vr_driving_pin(VRPin(net.get_driving_pin()));
  vr_net.set_bounding_box(net.get_bounding_box());
  vr_net.set_dr_result_tree(net.get_dr_result_tree());
  return vr_net;
}

void ViolationRepairer::buildVRModel(VRModel& vr_model)
{
  updateNetFixedRectMap(vr_model);
  calcVRGCellSupply(vr_model);
  updateVRResultTree(vr_model);
}

void ViolationRepairer::updateNetFixedRectMap(VRModel& vr_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = DM_INST.getDatabase().get_cut_blockage_list();

  for (Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    addRectToEnv(vr_model, VRSourceType::kLayoutShape, DRCRect(-1, blockage_real_rect, true));
  }
  for (Blockage& cut_blockage : cut_blockage_list) {
    LayerRect blockage_real_rect(cut_blockage.get_real_rect(), cut_blockage.get_layer_idx());
    addRectToEnv(vr_model, VRSourceType::kLayoutShape, DRCRect(-1, blockage_real_rect, false));
  }
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        addRectToEnv(vr_model, VRSourceType::kLayoutShape, DRCRect(vr_net.get_net_idx(), shape_real_rect, true));
      }
      for (EXTLayerRect& cut_shape : vr_pin.get_cut_shape_list()) {
        LayerRect shape_real_rect(cut_shape.get_real_rect(), cut_shape.get_layer_idx());
        addRectToEnv(vr_model, VRSourceType::kLayoutShape, DRCRect(vr_net.get_net_idx(), shape_real_rect, false));
      }
    }
  }
}

void ViolationRepairer::addRectToEnv(VRModel& vr_model, VRSourceType vr_source_type, DRCRect drc_rect)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();

  for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_rect)) {
    LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
    PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
    for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
      for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
        VRGCell& vr_gcell = vr_gcell_map[x][y];
        DC_INST.updateRectList(vr_gcell.getRegionQuery(vr_source_type), ChangeType::kAdd, drc_rect);
      }
    }
  }
}

void ViolationRepairer::calcVRGCellSupply(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();
// track supply
#pragma omp parallel for collapse(2)
  for (irt_int x = 0; x < vr_gcell_map.get_x_size(); x++) {
    for (irt_int y = 0; y < vr_gcell_map.get_y_size(); y++) {
      VRGCell& vr_gcell = vr_gcell_map[x][y];

      for (RoutingLayer& routing_layer : routing_layer_list) {
        irt_int whole_via_demand = routing_layer.get_min_area() / routing_layer.get_min_width();
        std::vector<PlanarRect> wire_list = getWireList(vr_gcell, routing_layer);
        if (!wire_list.empty()) {
          irt_int real_whole_wire_demand = wire_list.front().getArea() / routing_layer.get_min_width();
          irt_int gcell_whole_wire_demand = 0;
          if (routing_layer.isPreferH()) {
            gcell_whole_wire_demand = vr_gcell.get_base_region().getXSpan();
          } else {
            gcell_whole_wire_demand = vr_gcell.get_base_region().getYSpan();
          }
          if (real_whole_wire_demand != gcell_whole_wire_demand) {
            LOG_INST.error(Loc::current(), "The real_whole_wire_demand and gcell_whole_wire_demand are not equal!");
          }
        }
        for (VRSourceType vr_source_type : {VRSourceType::kLayoutShape}) {
          for (const auto& [net_idx, rect_set] :
               DC_INST.getLayerNetRectMap(vr_gcell.getRegionQuery(vr_source_type), true)[routing_layer.get_layer_idx()]) {
            for (const LayerRect& rect : rect_set) {
              for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCRect(net_idx, rect, true))) {
                std::vector<PlanarRect> new_wire_list;
                for (PlanarRect& wire : wire_list) {
                  if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
                    // 要切
                    std::vector<PlanarRect> split_rect_list
                        = RTUtil::getSplitRectList(wire, min_scope_real_rect, routing_layer.get_prefer_direction());
                    new_wire_list.insert(new_wire_list.end(), split_rect_list.begin(), split_rect_list.end());
                  } else {
                    // 不切
                    new_wire_list.push_back(wire);
                  }
                }
                wire_list = new_wire_list;
              }
            }
          }
        }
        for (PlanarRect& wire : wire_list) {
          irt_int supply = wire.getArea() / routing_layer.get_min_width();
          if (supply < whole_via_demand) {
            continue;
          }
          vr_gcell.get_layer_resource_supply_map()[routing_layer.get_layer_idx()] += supply;
        }
      }
    }
  }
}

std::vector<PlanarRect> ViolationRepairer::getWireList(VRGCell& vr_gcell, RoutingLayer& routing_layer)
{
  irt_int real_lb_x = vr_gcell.get_base_region().get_lb_x();
  irt_int real_lb_y = vr_gcell.get_base_region().get_lb_y();
  irt_int real_rt_x = vr_gcell.get_base_region().get_rt_x();
  irt_int real_rt_y = vr_gcell.get_base_region().get_rt_y();
  std::vector<irt_int> x_list = RTUtil::getOpenScaleList(real_lb_x, real_rt_x, routing_layer.getXTrackGridList());
  std::vector<irt_int> y_list = RTUtil::getOpenScaleList(real_lb_y, real_rt_y, routing_layer.getYTrackGridList());
  irt_int half_width = routing_layer.get_min_width() / 2;

  std::vector<PlanarRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (irt_int y : y_list) {
      wire_list.emplace_back(real_lb_x, y - half_width, real_rt_x, y + half_width);
    }
  } else {
    for (irt_int x : x_list) {
      wire_list.emplace_back(x - half_width, real_lb_y, x + half_width, real_rt_y);
    }
  }
  return wire_list;
}

void ViolationRepairer::updateVRResultTree(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    buildKeyCoordPinMap(vr_net);
    buildCoordTree(vr_net);
    buildPHYNodeResult(vr_net);
  }
}

void ViolationRepairer::buildKeyCoordPinMap(VRNet& vr_net)
{
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map = vr_net.get_key_coord_pin_map();
  for (Pin& vr_pin : vr_net.get_vr_pin_list()) {
    for (LayerCoord& real_coord : vr_pin.getRealCoordList()) {
      key_coord_pin_map[real_coord].insert(vr_pin.get_pin_idx());
    }
  }
}

void ViolationRepairer::buildCoordTree(VRNet& vr_net)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (TNode<RTNode>* rt_node_node : RTUtil::getNodeList(vr_net.get_dr_result_tree())) {
    for (Segment<TNode<LayerCoord>*>& routing_segment : RTUtil::getSegListByTree(rt_node_node->value().get_routing_tree())) {
      routing_segment_list.emplace_back(routing_segment.get_first()->value(), routing_segment.get_second()->value());
    }
  }
  std::vector<LayerCoord> candidate_root_coord_list;
  for (LayerCoord& real_coord : vr_net.get_vr_driving_pin().getRealCoordList()) {
    candidate_root_coord_list.push_back(real_coord);
  }
  MTree<LayerCoord>& coord_tree = vr_net.get_coord_tree();
  coord_tree = RTUtil::getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, vr_net.get_key_coord_pin_map());
}

void ViolationRepairer::buildPHYNodeResult(VRNet& vr_net)
{
  TNode<LayerCoord>* coord_root = vr_net.get_coord_tree().get_root();
  MTree<PHYNode>& vr_result_tree = vr_net.get_vr_result_tree();

  std::vector<TNode<PHYNode>*> pre_connection_list;
  std::vector<TNode<PHYNode>*> post_connection_list;
  updateConnectionList(coord_root, vr_net, pre_connection_list, post_connection_list);

  vr_result_tree.set_root(pre_connection_list.front());
  std::queue<TNode<LayerCoord>*> coord_node_queue = RTUtil::initQueue(coord_root->get_child_list());
  std::queue<TNode<PHYNode>*> phy_node_queue = RTUtil::initQueue(post_connection_list);

  while (!coord_node_queue.empty()) {
    TNode<LayerCoord>* coord_node = RTUtil::getFrontAndPop(coord_node_queue);
    TNode<PHYNode>* phy_node = RTUtil::getFrontAndPop(phy_node_queue);
    updateConnectionList(coord_node, vr_net, pre_connection_list, post_connection_list);
    phy_node->addChildren(pre_connection_list);
    RTUtil::addListToQueue(coord_node_queue, coord_node->get_child_list());
    RTUtil::addListToQueue(phy_node_queue, post_connection_list);
  }
}

void ViolationRepairer::updateConnectionList(TNode<LayerCoord>* coord_node, VRNet& vr_net,
                                             std::vector<TNode<PHYNode>*>& pre_connection_list,
                                             std::vector<TNode<PHYNode>*>& post_connection_list)
{
  pre_connection_list.clear();
  post_connection_list.clear();

  std::set<irt_int>& pin_idx_set = vr_net.get_key_coord_pin_map()[coord_node->value()];

  std::vector<irt_int> pin_idx_list;
  pin_idx_list.assign(pin_idx_set.begin(), pin_idx_set.end());
  if (!pin_idx_list.empty()) {
    TNode<PHYNode>* phy_node_node = makePinPHYNode(vr_net, pin_idx_list.front(), coord_node->value());
    pre_connection_list.push_back(phy_node_node);
    for (size_t i = 1; i < pin_idx_list.size(); i++) {
      phy_node_node->addChild(makePinPHYNode(vr_net, pin_idx_list[i], coord_node->value()));
    }
  }
  for (TNode<LayerCoord>* child_node : coord_node->get_child_list()) {
    LayerCoord& first_coord = coord_node->value();
    irt_int first_layer_idx = first_coord.get_layer_idx();
    LayerCoord& second_coord = child_node->value();
    irt_int second_layer_idx = second_coord.get_layer_idx();

    if (first_layer_idx == second_layer_idx) {
      post_connection_list.push_back(makeWirePHYNode(vr_net, first_coord, second_coord));
    } else {
      post_connection_list.push_back(makeViaPHYNode(vr_net, std::min(first_layer_idx, second_layer_idx), first_coord));
    }
  }
  if (pre_connection_list.empty()) {
    pre_connection_list = post_connection_list;
  } else if (pre_connection_list.size() == 1) {
    for (TNode<PHYNode>* post_connection : post_connection_list) {
      pre_connection_list.front()->addChild(post_connection);
    }
  } else {
    LOG_INST.error(Loc::current(), "The pre_connection_list size is ", pre_connection_list.size(), "!");
  }
}

TNode<PHYNode>* ViolationRepairer::makeWirePHYNode(VRNet& vr_net, LayerCoord first_coord, LayerCoord second_coord)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  if (RTUtil::isOblique(first_coord, second_coord)) {
    LOG_INST.error(Loc::current(), "The wire phy node is oblique!");
  }
  irt_int layer_idx = first_coord.get_layer_idx();
  if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
    LOG_INST.error(Loc::current(), "The wire layer_idx is illegal!");
  }
  RTUtil::swapByCMP(first_coord, second_coord, CmpLayerCoordByXASC());

  PHYNode phy_node;
  WireNode& wire_node = phy_node.getNode<WireNode>();
  wire_node.set_net_idx(vr_net.get_net_idx());
  wire_node.set_layer_idx(layer_idx);
  wire_node.set_first(first_coord);
  wire_node.set_second(second_coord);
  wire_node.set_wire_width(routing_layer_list[layer_idx].get_min_width());
  return (new TNode<PHYNode>(phy_node));
}

TNode<PHYNode>* ViolationRepairer::makeViaPHYNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  if (below_layer_idx < 0 || below_layer_idx >= static_cast<irt_int>(layer_via_master_list.size())) {
    LOG_INST.error(Loc::current(), "The via below_layer_idx is illegal!");
  }
  PHYNode phy_node;
  ViaNode& via_node = phy_node.getNode<ViaNode>();
  via_node.set_net_idx(vr_net.get_net_idx());
  via_node.set_coord(coord);
  via_node.set_via_master_idx(layer_via_master_list[below_layer_idx].front().get_via_master_idx());
  return (new TNode<PHYNode>(phy_node));
}

TNode<PHYNode>* ViolationRepairer::makePinPHYNode(VRNet& vr_net, irt_int pin_idx, LayerCoord coord)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  irt_int layer_idx = coord.get_layer_idx();
  if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
    LOG_INST.error(Loc::current(), "The pin layer_idx is illegal!");
  }
  PHYNode phy_node;
  PinNode& pin_node = phy_node.getNode<PinNode>();
  pin_node.set_net_idx(vr_net.get_net_idx());
  pin_node.set_pin_idx(pin_idx);
  pin_node.set_coord(coord);
  pin_node.set_layer_idx(layer_idx);
  return (new TNode<PHYNode>(phy_node));
}

void ViolationRepairer::checkVRModel(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    if (vr_net.get_net_idx() < 0) {
      LOG_INST.error(Loc::current(), "The net idx : ", vr_net.get_net_idx(), " is illegal!");
    }
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      std::vector<AccessPoint>& access_point_list = vr_pin.get_access_point_list();
      if (access_point_list.empty()) {
        LOG_INST.error(Loc::current(), "The pin ", vr_pin.get_pin_idx(), " access point list is empty!");
      }
      for (AccessPoint& access_point : access_point_list) {
        if (access_point.get_type() == AccessPointType::kNone) {
          LOG_INST.error(Loc::current(), "The access point type is wrong!");
        }
        bool is_legal = false;
        for (EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
          if (routing_shape.get_layer_idx() == access_point.get_layer_idx()
              && RTUtil::isInside(routing_shape.get_real_rect(), access_point.get_real_coord())) {
            is_legal = true;
            break;
          }
        }
        if (!is_legal) {
          LOG_INST.error(Loc::current(), "The access point is not in routing shape!");
        }
      }
    }
  }
}

#endif

#if 1  // iterative

void ViolationRepairer::iterative(VRModel& vr_model)
{
  irt_int vr_max_iter_num = DM_INST.getConfig().vr_max_iter_num;

  for (irt_int iter = 1; iter <= vr_max_iter_num; iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Iteration(", iter, "/", vr_max_iter_num, ") ******");
    vr_model.set_curr_iter(iter);
    repairVRModel(vr_model);
    processVRModel(vr_model);
    countVRModel(vr_model);
    reportVRModel(vr_model);
    LOG_INST.info(Loc::current(), "****** End Iteration(", iter, "/", vr_max_iter_num, ")", iter_monitor.getStatsInfo(), " ******");
    if (stopVRModel(vr_model)) {
      LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      vr_model.set_curr_iter(-1);
      break;
    }
  }
}

void ViolationRepairer::repairVRModel(VRModel& vr_model)
{
  repairAntenna(vr_model);
  repairMinStep(vr_model);
  repairMinArea(vr_model);
}

void ViolationRepairer::repairAntenna(VRModel& vr_model)
{
}

void ViolationRepairer::repairMinStep(VRModel& vr_model)
{
}

void ViolationRepairer::repairMinArea(VRModel& vr_model)
{
  Monitor monitor;

  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

  irt_int batch_size = RTUtil::getBatchSize(vr_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < vr_net_list.size(); i++) {
    repairMinArea(vr_model, vr_net_list[i]);
    if ((i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Repaired ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  LOG_INST.info(Loc::current(), "Repaired ", vr_net_list.size(), " nets", monitor.getStatsInfo());
}

void ViolationRepairer::repairMinArea(VRModel& vr_model, VRNet& vr_net)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::map<irt_int, gtl::polygon_90_set_data<irt_int>> layer_polygon_set_map;
  {
    // pin_shape
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (const EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        layer_polygon_set_map[shape_real_rect.get_layer_idx()] += RTUtil::convertToGTLRect(shape_real_rect);
      }
    }
    // vr_result_tree
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      if (!drc_rect.get_is_routing()) {
        continue;
      }
      layer_polygon_set_map[drc_rect.get_layer_idx()] += RTUtil::convertToGTLRect(drc_rect.get_layer_rect());
    }
  }
  std::map<LayerRect, irt_int, CmpLayerRectByXASC> violation_rect_added_area_map;
  for (auto& [layer_idx, polygon_set] : layer_polygon_set_map) {
    irt_int layer_min_area = routing_layer_list[layer_idx].get_min_area();
    std::vector<gtl::polygon_90_data<irt_int>> polygon_list;
    polygon_set.get_polygons(polygon_list);
    for (gtl::polygon_90_data<irt_int>& polygon : polygon_list) {
      if (gtl::area(polygon) >= layer_min_area) {
        continue;
      }
      // 取polygon中最大的矩形进行膨胀
      PlanarRect max_violation_rect;
      std::vector<gtl::rectangle_data<irt_int>> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, polygon);
      for (gtl::rectangle_data<irt_int>& gtl_rect : gtl_rect_list) {
        if (max_violation_rect.getArea() < gtl::area(gtl_rect)) {
          max_violation_rect = RTUtil::convertToPlanarRect(gtl_rect);
        }
      }
      irt_int added_area = layer_min_area - gtl::area(polygon);
      violation_rect_added_area_map[LayerRect(max_violation_rect, layer_idx)] = added_area;
    }
  }
  std::vector<LayerRect> patch_list;
  for (auto& [violation_rect, added_area] : violation_rect_added_area_map) {
    irt_int layer_idx = violation_rect.get_layer_idx();

    std::vector<LayerRect> h_candidate_patch_list;
    irt_int h_enlarged_offset = static_cast<irt_int>(std::ceil(added_area / 1.0 / violation_rect.getYSpan()));
    for (irt_int lb_offset = 0; lb_offset <= h_enlarged_offset; lb_offset++) {
      h_candidate_patch_list.emplace_back(
          RTUtil::getEnlargedRect(violation_rect, lb_offset, 0, h_enlarged_offset - lb_offset, 0, die.get_real_rect()), layer_idx);
    }
    std::vector<LayerRect> v_candidate_patch_list;
    irt_int v_enlarged_offset = static_cast<irt_int>(std::ceil(added_area / 1.0 / violation_rect.getXSpan()));
    for (irt_int lb_offset = 0; lb_offset <= v_enlarged_offset; lb_offset++) {
      v_candidate_patch_list.emplace_back(
          RTUtil::getEnlargedRect(violation_rect, 0, lb_offset, 0, v_enlarged_offset - lb_offset, die.get_real_rect()), layer_idx);
    }

    std::vector<LayerRect> candidate_patch_list;
    if (routing_layer_list[layer_idx].isPreferH()) {
      candidate_patch_list.insert(candidate_patch_list.end(), h_candidate_patch_list.begin(), h_candidate_patch_list.end());
      candidate_patch_list.insert(candidate_patch_list.end(), v_candidate_patch_list.begin(), v_candidate_patch_list.end());
    } else {
      candidate_patch_list.insert(candidate_patch_list.end(), v_candidate_patch_list.begin(), v_candidate_patch_list.end());
      candidate_patch_list.insert(candidate_patch_list.end(), h_candidate_patch_list.begin(), h_candidate_patch_list.end());
    }
    for (LayerRect& candidate_patch : candidate_patch_list) {
      if (!hasViolation(vr_model, VRSourceType::kLayoutShape, DRCRect(vr_net.get_net_idx(), candidate_patch, true))) {
        patch_list.push_back(candidate_patch);
        break;
      }
    }
  }
  for (LayerRect& patch : patch_list) {
    TNode<PHYNode>* root_node = vr_net.get_vr_result_tree().get_root();

    PHYNode phy_node;
    PatchNode& patch_node = phy_node.getNode<PatchNode>();
    patch_node.set_net_idx(vr_net.get_net_idx());
    patch_node.set_rect(patch);
    patch_node.set_layer_idx(patch.get_layer_idx());

    root_node->addChild(new TNode<PHYNode>(phy_node));
  }
}

void ViolationRepairer::processVRModel(VRModel& vr_model)
{
  updateNetResultMap(vr_model);
  calcVRGCellDemand(vr_model);
}

void ViolationRepairer::updateNetResultMap(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      addRectToEnv(vr_model, VRSourceType::kLayoutShape, drc_rect);
    }
  }
}

void ViolationRepairer::calcVRGCellDemand(VRModel& vr_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();

  std::map<LayerCoord, std::vector<LayerRect>, CmpLayerCoordByXASC> grid_net_result_map;
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      LayerRect& layer_rect = drc_rect.get_layer_rect();
      PlanarRect grid_rect = RTUtil::getClosedGridRect(layer_rect, gcell_axis);
      for (irt_int x = grid_rect.get_lb_x(); x <= grid_rect.get_rt_x(); x++) {
        for (irt_int y = grid_rect.get_lb_y(); y <= grid_rect.get_rt_y(); y++) {
          grid_net_result_map[LayerCoord(x, y, layer_rect.get_layer_idx())].push_back(layer_rect);
        }
      }
    }
  }
  for (auto& [grid, rect_list] : grid_net_result_map) {
    irt_int grid_layer_idx = grid.get_layer_idx();
    RoutingLayer& routing_layer = routing_layer_list[grid_layer_idx];
    VRGCell& vr_gcell = vr_gcell_map[grid.get_x()][grid.get_y()];
    PlanarRect& base_region = vr_gcell.get_base_region();

    irt_int demand = 0;
    for (LayerRect& rect : rect_list) {
      if (!RTUtil::isClosedOverlap(rect, base_region)) {
        continue;
      }
      PlanarRect overlap_rect = RTUtil::getOverlap(rect, base_region);
      demand += (overlap_rect.getArea() / routing_layer.get_min_width());
    }
    vr_gcell.get_layer_resource_demand_map()[grid_layer_idx] += demand;
  }
}

void ViolationRepairer::countVRModel(VRModel& vr_model)
{
  irt_int micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  VRModelStat vr_model_stat;

  std::map<irt_int, double>& routing_wire_length_map = vr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, double>& routing_prefer_wire_length_map = vr_model_stat.get_routing_prefer_wire_length_map();
  std::map<irt_int, double>& routing_nonprefer_wire_length_map = vr_model_stat.get_routing_nonprefer_wire_length_map();
  std::map<irt_int, irt_int>& routing_patch_number_map = vr_model_stat.get_routing_patch_number_map();
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();

  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(vr_net.get_vr_result_tree())) {
      PHYNode& phy_node = phy_node_node->value();
      if (phy_node.isType<WireNode>()) {
        WireNode& wire_node = phy_node.getNode<WireNode>();
        irt_int layer_idx = wire_node.get_layer_idx();
        double wire_length = RTUtil::getManhattanDistance(wire_node.get_first(), wire_node.get_second()) / 1.0 / micron_dbu;
        if (RTUtil::getDirection(wire_node.get_first(), wire_node.get_second()) == routing_layer_list[layer_idx].get_prefer_direction()) {
          routing_prefer_wire_length_map[layer_idx] += wire_length;
        } else {
          routing_nonprefer_wire_length_map[layer_idx] += wire_length;
        }
        routing_wire_length_map[wire_node.get_layer_idx()] += wire_length;
      } else if (phy_node.isType<ViaNode>()) {
        ViaNode& via_node = phy_node.getNode<ViaNode>();
        cut_via_number_map[via_node.get_via_master_idx().get_below_layer_idx()]++;
      } else if (phy_node.isType<PatchNode>()) {
        PatchNode& patch_node = phy_node.getNode<PatchNode>();
        routing_patch_number_map[patch_node.get_layer_idx()]++;
      }
    }
  }

  std::map<irt_int, std::vector<double>>& layer_resource_overflow_map = vr_model_stat.get_layer_resource_overflow_map();

  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();
  for (irt_int x = 0; x < vr_gcell_map.get_x_size(); x++) {
    for (irt_int y = 0; y < vr_gcell_map.get_y_size(); y++) {
      VRGCell& vr_gcell = vr_gcell_map[x][y];
      std::map<irt_int, irt_int>& layer_resource_supply_map = vr_gcell.get_layer_resource_supply_map();
      std::map<irt_int, irt_int>& layer_resource_demand_map = vr_gcell.get_layer_resource_demand_map();

      for (RoutingLayer& routing_layer : routing_layer_list) {
        irt_int layer_idx = routing_layer.get_layer_idx();
        irt_int supply = 0;
        if (RTUtil::exist(layer_resource_supply_map, layer_idx)) {
          supply = layer_resource_supply_map[layer_idx];
        }
        irt_int demand = 0;
        if (RTUtil::exist(layer_resource_demand_map, layer_idx)) {
          demand = layer_resource_demand_map[layer_idx];
        }
        layer_resource_overflow_map[layer_idx].push_back(RTUtil::calcCost(demand, supply));
      }
    }
  }

  std::map<VRSourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = vr_model_stat.get_source_drc_violation_map();
  for (irt_int x = 0; x < vr_gcell_map.get_x_size(); x++) {
    for (irt_int y = 0; y < vr_gcell_map.get_y_size(); y++) {
      VRGCell& vr_gcell = vr_gcell_map[x][y];

      for (VRSourceType vr_source_type : {VRSourceType::kLayoutShape}) {
        for (auto& [drc, violation_info_list] : getViolationInfo(vr_gcell, vr_source_type)) {
          std::vector<ViolationInfo>& total_violation_list = source_drc_violation_map[vr_source_type][drc];
          total_violation_list.insert(total_violation_list.end(), violation_info_list.begin(), violation_info_list.end());
        }
      }
    }
  }

  std::map<std::string, irt_int>& rule_number_map = vr_model_stat.get_drc_number_map();
  for (auto& [vr_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      rule_number_map[drc] += violation_list.size();
    }
  }
  std::map<std::string, irt_int>& source_number_map = vr_model_stat.get_source_number_map();
  for (auto& [vr_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int total_number = 0;
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_number += violation_list.size();
    }
    source_number_map[GetVRSourceTypeName()(vr_source_type)] = total_number;
  }

  double total_wire_length = 0;
  double total_prefer_wire_length = 0;
  double total_nonprefer_wire_length = 0;
  irt_int total_patch_number = 0;
  irt_int total_via_number = 0;
  irt_int total_overflow_number = 0;
  irt_int total_drc_number = 0;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    total_wire_length += wire_length;
  }
  for (auto& [routing_layer_idx, prefer_wire_length] : routing_prefer_wire_length_map) {
    total_prefer_wire_length += prefer_wire_length;
  }
  for (auto& [routing_layer_idx, nonprefer_wire_length] : routing_nonprefer_wire_length_map) {
    total_nonprefer_wire_length += nonprefer_wire_length;
  }
  for (auto& [routing_layer_idx, patch_number] : routing_patch_number_map) {
    total_patch_number += patch_number;
  }
  for (auto& [cut_layer_idx, via_number] : cut_via_number_map) {
    total_via_number += via_number;
  }
  for (auto& [layer_idx, resource_overflow_list] : layer_resource_overflow_map) {
    total_overflow_number += resource_overflow_list.size();
  }
  for (auto& [vr_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_drc_number += violation_list.size();
    }
  }
  vr_model_stat.set_total_wire_length(total_wire_length);
  vr_model_stat.set_total_prefer_wire_length(total_prefer_wire_length);
  vr_model_stat.set_total_nonprefer_wire_length(total_nonprefer_wire_length);
  vr_model_stat.set_total_patch_number(total_patch_number);
  vr_model_stat.set_total_via_number(total_via_number);
  vr_model_stat.set_total_overflow_number(total_overflow_number);
  vr_model_stat.set_total_drc_number(total_drc_number);

  vr_model.set_vr_model_stat(vr_model_stat);
}

void ViolationRepairer::reportVRModel(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  // wire table
  VRModelStat& vr_model_stat = vr_model.get_vr_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = vr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, double>& routing_prefer_wire_length_map = vr_model_stat.get_routing_prefer_wire_length_map();
  std::map<irt_int, double>& routing_nonprefer_wire_length_map = vr_model_stat.get_routing_nonprefer_wire_length_map();
  std::map<irt_int, irt_int>& routing_patch_number_map = vr_model_stat.get_routing_patch_number_map();
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();
  std::map<irt_int, std::vector<double>>& layer_resource_overflow_map = vr_model_stat.get_layer_resource_overflow_map();
  std::map<VRSourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = vr_model_stat.get_source_drc_violation_map();
  std::map<std::string, irt_int>& rule_number_map = vr_model_stat.get_drc_number_map();
  std::map<std::string, irt_int>& source_number_map = vr_model_stat.get_source_number_map();
  double total_wire_length = vr_model_stat.get_total_wire_length();
  double total_prefer_wire_length = vr_model_stat.get_total_prefer_wire_length();
  double total_nonprefer_wire_length = vr_model_stat.get_total_nonprefer_wire_length();
  irt_int total_patch_number = vr_model_stat.get_total_patch_number();
  irt_int total_via_number = vr_model_stat.get_total_via_number();
  irt_int total_overflow_number = vr_model_stat.get_total_overflow_number();
  irt_int total_drc_number = vr_model_stat.get_total_drc_number();

  // wire table
  fort::char_table wire_table;
  wire_table << fort::header << "Routing Layer"
             << "Prefer Wire Length"
             << "Nonprefer Wire Length"
             << "Wire Length / um"
             << "Patch Number" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_idx = routing_layer.get_layer_idx();
    wire_table << routing_layer.get_layer_name() << routing_prefer_wire_length_map[layer_idx]
               << routing_nonprefer_wire_length_map[layer_idx] << routing_wire_length_map[layer_idx] << routing_patch_number_map[layer_idx]
               << fort::endr;
  }
  wire_table << fort::header << "Total" << total_prefer_wire_length << total_nonprefer_wire_length << total_wire_length
             << total_patch_number << fort::endr;
  // via table
  fort::char_table via_table;
  via_table << fort::header << "Cut Layer"
            << "Via Number" << fort::endr;
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int cut_via_number = cut_via_number_map[cut_layer.get_layer_idx()];
    via_table << cut_layer.get_layer_name()
              << RTUtil::getString(cut_via_number, "(", RTUtil::getPercentage(cut_via_number, total_via_number), "%)") << fort::endr;
  }
  via_table << fort::header << "Total" << total_via_number << fort::endr;
  // patch table
  fort::char_table patch_table;
  patch_table << fort::header << "Routing Layer"
              << "Patch Number" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_idx = routing_layer.get_layer_idx();
    irt_int patch_number = routing_patch_number_map[layer_idx];
    patch_table << routing_layer.get_layer_name()
                << RTUtil::getString(patch_number, "(", RTUtil::getPercentage(patch_number, total_patch_number), "%)") << fort::endr;
  }
  patch_table << fort::header << "Total" << total_patch_number << fort::endr;

  printTableList({wire_table, via_table, patch_table});

  // report resource overflow info
  auto layer_resource_range_number_map = RTUtil::getLayerRangeNumMap(layer_resource_overflow_map, {1.0});
  fort::char_table resource_overflow_table = buildOverflowTable(layer_resource_range_number_map, total_overflow_number);
  resource_overflow_table[0][0] = "Layer\\Resource Overflow";

  // init item column/row map
  irt_int row = 0;
  std::map<std::string, irt_int> item_row_map;
  for (auto& [drc_rule, drc_number] : rule_number_map) {
    item_row_map[drc_rule] = ++row;
  }
  item_row_map["Total"] = ++row;

  irt_int column = 0;
  std::map<std::string, irt_int> item_column_map;
  for (auto& [vr_source_type, drc_number_map] : source_number_map) {
    item_column_map[vr_source_type] = ++column;
  }
  item_column_map["Total"] = ++column;

  // build table
  fort::char_table drc_table;
  drc_table << fort::header;
  drc_table[0][0] = "DRC\\Source";
  // first row item
  for (auto& [drc_rule, row] : item_row_map) {
    drc_table[row][0] = drc_rule;
  }
  // first column item
  drc_table << fort::header;
  for (auto& [source_name, column] : item_column_map) {
    drc_table[0][column] = source_name;
  }
  // element
  for (auto& [vr_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int column = item_column_map[GetVRSourceTypeName()(vr_source_type)];
    for (auto& [drc_rule, row] : item_row_map) {
      if (RTUtil::exist(source_drc_violation_map[vr_source_type], drc_rule)) {
        drc_table[row][column] = RTUtil::getString(source_drc_violation_map[vr_source_type][drc_rule].size());
      } else {
        drc_table[row][column] = "0";
      }
    }
  }
  // last row
  for (auto& [vr_source_type, total_number] : source_number_map) {
    irt_int row = item_row_map["Total"];
    irt_int column = item_column_map[vr_source_type];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  // last column
  for (auto& [drc_rule, total_number] : rule_number_map) {
    irt_int row = item_row_map[drc_rule];
    irt_int column = item_column_map["Total"];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  drc_table[item_row_map["Total"]][item_column_map["Total"]] = RTUtil::getString(total_drc_number);

  // print
  printTableList({resource_overflow_table, drc_table});
}

fort::char_table ViolationRepairer::buildOverflowTable(
    std::map<irt_int, std::map<std::pair<double, double>, irt_int>>& layer_range_number_map, irt_int total_overflow_number)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  // init resource overflow table item column/row map
  irt_int report_number = 0;
  std::map<irt_int, irt_int> resource_layer_number_map;
  std::map<std::pair<double, double>, irt_int> resource_range_number_map;
  for (auto& [layer_idx, range_number_map] : layer_range_number_map) {
    irt_int layer_total_number = 0;
    for (auto& [range, number] : range_number_map) {
      layer_total_number += number;
    }
    report_number += layer_total_number;
    resource_layer_number_map[layer_idx] = layer_total_number;
  }
  for (auto& [layer_idx, range_number_map] : layer_range_number_map) {
    for (auto& [range, number] : range_number_map) {
      resource_range_number_map[range] += number;
    }
  }

  std::map<std::pair<double, double>, std::string> range_str_map;
  for (auto& [range, number] : resource_range_number_map) {
    range_str_map[range] = RTUtil::getString("(", range.first, ",", range.second, ")");
  }

  irt_int row = 0;
  std::map<std::string, irt_int> item_row_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    item_row_map[routing_layer.get_layer_name()] = ++row;
  }
  item_row_map["Total"] = ++row;

  irt_int column = 0;
  std::map<std::string, irt_int> item_column_map;
  for (auto& [range, number] : resource_range_number_map) {
    item_column_map[range_str_map[range]] = ++column;
  }
  item_column_map["Total"] = ++column;

  // report resource overflow info
  fort::char_table resource_overflow_table;
  resource_overflow_table << fort::header << "Layer\\Overflow" << fort::endr;
  for (auto& [range, number] : resource_range_number_map) {
    resource_overflow_table << range_str_map[range];
  }
  resource_overflow_table << fort::endr;

  // first row item
  for (auto& [layer_name, row] : item_row_map) {
    resource_overflow_table[row][0] = layer_name;
  }
  // first column item
  for (auto& [range_str, column] : item_column_map) {
    resource_overflow_table[0][column] = range_str;
  }
  // element
  for (auto& [layer, range_number_map] : layer_range_number_map) {
    irt_int row = item_row_map[routing_layer_list[layer].get_layer_name()];
    for (auto& [range, number] : range_number_map) {
      irt_int column = item_column_map[range_str_map[range]];
      resource_overflow_table[row][column] = RTUtil::getString(number, "(", RTUtil::getPercentage(number, total_overflow_number), "%)");
    }
  }
  // last row
  for (auto& [resource_range, total_number] : resource_range_number_map) {
    irt_int row = item_row_map["Total"];
    irt_int column = item_column_map[range_str_map[resource_range]];
    resource_overflow_table[row][column]
        = RTUtil::getString(total_number, "(", RTUtil::getPercentage(total_number, total_overflow_number), "%)");
  }
  resource_overflow_table << fort::header;

  // last column
  for (auto& [layer, total_number] : resource_layer_number_map) {
    irt_int row = item_row_map[routing_layer_list[layer].get_layer_name()];
    irt_int column = item_column_map["Total"];
    resource_overflow_table[row][column]
        = RTUtil::getString(total_number, "(", RTUtil::getPercentage(total_number, total_overflow_number), "%)");
  }

  resource_overflow_table[item_row_map["Total"]][item_column_map["Total"]]
      = RTUtil::getString(report_number, "(", RTUtil::getPercentage(report_number, total_overflow_number), "%)");

  return resource_overflow_table;
}

void ViolationRepairer::printTableList(const std::vector<fort::char_table>& table_list)
{
  std::vector<std::vector<std::string>> print_table_list;
  for (const fort::char_table& table : table_list) {
    print_table_list.push_back(RTUtil::splitString(table.to_string(), '\n'));
  }

  int max_size = INT_MIN;
  for (std::vector<std::string>& table : print_table_list) {
    max_size = std::max(max_size, static_cast<irt_int>(table.size()));
  }
  for (std::vector<std::string>& table : print_table_list) {
    for (irt_int i = table.size(); i < max_size; i++) {
      std::string table_str;
      table_str.append(table.front().length(), ' ');
      table.push_back(table_str);
    }
  }

  for (irt_int i = 0; i < max_size; i++) {
    std::string table_str;
    for (std::vector<std::string>& table : print_table_list) {
      table_str += table[i];
      table_str += " ";
    }
    LOG_INST.info(Loc::current(), table_str);
  }
}

bool ViolationRepairer::stopVRModel(VRModel& vr_model)
{
  return (vr_model.get_vr_model_stat().get_total_drc_number() == 0);
}

#endif

#if 1  // update

void ViolationRepairer::update(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    Net* origin_net = vr_net.get_origin_net();
    origin_net->set_vr_result_tree(vr_net.get_vr_result_tree());
  }
}

#endif

#if 1  // valid drc

bool ViolationRepairer::hasViolation(VRModel& vr_model, VRSourceType vr_source_type, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list = {drc_rect};
  return hasViolation(vr_model, vr_source_type, drc_rect_list);
}

bool ViolationRepairer::hasViolation(VRModel& vr_model, VRSourceType vr_source_type, const std::vector<DRCRect>& drc_rect_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();

  std::map<VRGCellId, std::vector<DRCRect>, CmpVRGCellId> gcell_rect_map;
  for (const DRCRect& drc_rect : drc_rect_list) {
    for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_rect)) {
      PlanarRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
      PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
      for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
        for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
          gcell_rect_map[VRGCellId(x, y)].push_back(drc_rect);
        }
      }
    }
  }
  bool has_violation = false;
  for (const auto& [vr_gcell_id, drc_rect_list] : gcell_rect_map) {
    VRGCell& vr_gcell = vr_gcell_map[vr_gcell_id.get_x()][vr_gcell_id.get_y()];
    if (getViolationInfo(vr_gcell, vr_source_type, drc_rect_list).size() > 0) {
      has_violation = true;
      break;
    }
  }
  return has_violation;
}

std::map<std::string, std::vector<ViolationInfo>> ViolationRepairer::getViolationInfo(VRGCell& vr_gcell, VRSourceType vr_source_type,
                                                                                      const std::vector<DRCRect>& drc_rect_list)
{
  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  drc_violation_map = DC_INST.getViolationInfo(vr_gcell.getRegionQuery(vr_source_type), drc_rect_list);
  removeInvalidViolationInfo(vr_gcell, drc_violation_map);
  return drc_violation_map;
}

std::map<std::string, std::vector<ViolationInfo>> ViolationRepairer::getViolationInfo(VRGCell& vr_gcell, VRSourceType vr_source_type)
{
  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  drc_violation_map = DC_INST.getViolationInfo(vr_gcell.getRegionQuery(vr_source_type));
  removeInvalidViolationInfo(vr_gcell, drc_violation_map);
  return drc_violation_map;
}

void ViolationRepairer::removeInvalidViolationInfo(VRGCell& vr_gcell, std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map)
{
  for (auto& [drc, violation_list] : drc_violation_map) {
    std::vector<ViolationInfo> valid_violation_list;
    for (ViolationInfo& violation_info : violation_list) {
      bool is_valid = false;
      for (auto& [net_idx, rect_list] : violation_info.get_net_shape_map()) {
        if (net_idx != -1) {
          is_valid = true;
          break;
        }
      }
      if (is_valid) {
        valid_violation_list.push_back(violation_info);
      }
    }
    if (valid_violation_list.empty()) {
      drc_violation_map.erase(drc);
    } else {
      drc_violation_map[drc] = violation_list;
    }
  }
}

#endif

}  // namespace irt
