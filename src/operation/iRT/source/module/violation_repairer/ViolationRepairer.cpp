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

#include "DRCChecker.hpp"

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
  updateVRResultTree(vr_model);
}

void ViolationRepairer::updateNetFixedRectMap(VRModel& vr_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = DM_INST.getDatabase().get_cut_blockage_list();

  for (Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    addRectToEnv(vr_model, VRSourceType::kBlockAndPin, DRCRect(-1, blockage_real_rect, true));
  }
  for (Blockage& cut_blockage : cut_blockage_list) {
    LayerRect blockage_real_rect(cut_blockage.get_real_rect(), cut_blockage.get_layer_idx());
    addRectToEnv(vr_model, VRSourceType::kBlockAndPin, DRCRect(-1, blockage_real_rect, false));
  }
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        addRectToEnv(vr_model, VRSourceType::kBlockAndPin, DRCRect(vr_net.get_net_idx(), shape_real_rect, true));
      }
      for (EXTLayerRect& cut_shape : vr_pin.get_cut_shape_list()) {
        LayerRect shape_real_rect(cut_shape.get_real_rect(), cut_shape.get_layer_idx());
        addRectToEnv(vr_model, VRSourceType::kBlockAndPin, DRCRect(vr_net.get_net_idx(), shape_real_rect, false));
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
        DC_INST.addEnvRectList(vr_gcell.getRegionQuery(vr_source_type), drc_rect);
      }
    }
  }
}

void ViolationRepairer::updateVRResultTree(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    buildKeyCoordPinMap(vr_net);
    buildCoordTree(vr_net);
    buildPHYNodeResult(vr_net);
    updateVRGCellMap(vr_model, vr_net);
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

void ViolationRepairer::updateVRGCellMap(VRModel& vr_model, VRNet& vr_net)
{
  for (DRCRect& drc_rect : DC_INST.getDRCRectList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
    addRectToEnv(vr_model, VRSourceType::kNetResult, drc_rect);
  }
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
}

void ViolationRepairer::countVRModel(VRModel& vr_model)
{
  irt_int micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  VRModelStat vr_model_stat;

  std::map<irt_int, double>& routing_wire_length_map = vr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, double>& routing_prefer_wire_length_map = vr_model_stat.get_routing_prefer_wire_length_map();
  std::map<irt_int, double>& routing_nonprefer_wire_length_map = vr_model_stat.get_routing_nonprefer_wire_length_map();
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();

  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(vr_net.get_vr_result_tree())) {
      PHYNode& phy_node = phy_node_node->value();
      if (phy_node.isType<WireNode>()) {
        WireNode& wire_node = phy_node.getNode<WireNode>();
        irt_int layer_idx = wire_node.get_layer_idx();
        double wire_length = RTUtil::getManhattanDistance(wire_node.get_first(), wire_node.get_second()) / 1.0 / micron_dbu;
        if (RTUtil::getDirection(wire_node.get_first(), wire_node.get_second()) == routing_layer_list[layer_idx].get_direction()) {
          routing_prefer_wire_length_map[layer_idx] += wire_length;
        } else {
          routing_nonprefer_wire_length_map[layer_idx] += wire_length;
        }
        routing_wire_length_map[wire_node.get_layer_idx()] += wire_length;
      } else if (phy_node.isType<ViaNode>()) {
        ViaNode& via_node = phy_node.getNode<ViaNode>();
        cut_via_number_map[via_node.get_via_master_idx().get_below_layer_idx()]++;
      }
    }
  }

  std::map<VRSourceType, std::map<std::string, irt_int>>& source_drc_number_map = vr_model_stat.get_source_drc_number_map();
  GridMap<VRGCell>& vr_gcell_map = vr_model.get_vr_gcell_map();
  for (irt_int x = 0; x < vr_gcell_map.get_x_size(); x++) {
    for (irt_int y = 0; y < vr_gcell_map.get_y_size(); y++) {
      VRGCell& vr_gcell = vr_gcell_map[x][y];

      std::vector<DRCRect> drc_rect_list;
      for (bool is_routing : {true, false}) {
        for (auto& [layer_idx, net_rect_map] : DC_INST.getLayerNetRectMap(vr_gcell.getRegionQuery(VRSourceType::kNetResult), is_routing)) {
          for (auto& [net_idx, rect_set] : net_rect_map) {
            for (const LayerRect& layer_rect : rect_set) {
              drc_rect_list.emplace_back(net_idx, layer_rect, is_routing);
            }
          }
        }
      }

      for (VRSourceType vr_source_type : {VRSourceType::kBlockAndPin, VRSourceType::kNetResult}) {
        RegionQuery* region_query = vr_gcell.getRegionQuery(vr_source_type);
        for (auto& [drc, number] : DC_INST.getViolation(region_query, drc_rect_list)) {
          source_drc_number_map[vr_source_type][drc] += number;
        }
      }
    }
  }

  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    std::vector<DRCRect> drc_rect_list;
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      drc_rect_list.push_back(drc_rect);
    }
    for (auto& [drc, number] : DC_INST.getViolation(drc_rect_list)) {
      source_drc_number_map[VRSourceType::kNetResult][drc] += number;
    }
  }

  std::map<std::string, irt_int>& rule_number_map = vr_model_stat.get_drc_number_map();
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    for (auto& [drc, number] : drc_number_map) {
      rule_number_map[drc] += number;
    }
  }

  std::map<std::string, irt_int>& source_number_map = vr_model_stat.get_source_number_map();
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    irt_int total_number = 0;
    for (auto& [drc, number] : drc_number_map) {
      total_number += number;
    }
    source_number_map[GetVRSourceTypeName()(source)] = total_number;
  }

  double total_wire_length = 0;
  double total_prefer_wire_length = 0;
  double total_nonprefer_wire_length = 0;
  irt_int total_via_number = 0;
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
  for (auto& [cut_layer_idx, via_number] : cut_via_number_map) {
    total_via_number += via_number;
  }
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    for (auto& [drc, number] : drc_number_map) {
      total_drc_number += number;
    }
  }
  vr_model_stat.set_total_wire_length(total_wire_length);
  vr_model_stat.set_total_prefer_wire_length(total_prefer_wire_length);
  vr_model_stat.set_total_nonprefer_wire_length(total_nonprefer_wire_length);
  vr_model_stat.set_total_via_number(total_via_number);
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
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();
  std::map<VRSourceType, std::map<std::string, irt_int>>& source_drc_number_map = vr_model_stat.get_source_drc_number_map();
  std::map<std::string, irt_int>& rule_number_map = vr_model_stat.get_drc_number_map();
  std::map<std::string, irt_int>& source_number_map = vr_model_stat.get_source_number_map();
  double total_wire_length = vr_model_stat.get_total_wire_length();
  double total_prefer_wire_length = vr_model_stat.get_total_prefer_wire_length();
  double total_nonprefer_wire_length = vr_model_stat.get_total_nonprefer_wire_length();
  irt_int total_via_number = vr_model_stat.get_total_via_number();
  irt_int total_drc_number = vr_model_stat.get_total_drc_number();

  // wire table
  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_ROUND_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Prefer Wire Length"
             << "Nonprefer Wire Length"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_idx = routing_layer.get_layer_idx();
    wire_table << routing_layer.get_layer_name() << routing_prefer_wire_length_map[layer_idx]
               << routing_nonprefer_wire_length_map[layer_idx] << routing_wire_length_map[layer_idx] << fort::endr;
  }
  wire_table << fort::header << "Total" << total_prefer_wire_length << total_nonprefer_wire_length << total_wire_length << fort::endr;
  // via table
  fort::char_table via_table;
  via_table.set_border_style(FT_SOLID_ROUND_STYLE);
  via_table << fort::header << "Cut Layer"
            << "Via number" << fort::endr;
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int cut_via_number = cut_via_number_map[cut_layer.get_layer_idx()];
    via_table << cut_layer.get_layer_name()
              << RTUtil::getString(cut_via_number, "(", RTUtil::getPercentage(cut_via_number, total_via_number), "%)") << fort::endr;
  }
  via_table << fort::header << "Total" << total_via_number << fort::endr;

  // init item column/row map
  irt_int row = 0;
  std::map<std::string, irt_int> item_row_map;
  for (auto& [drc_rule, drc_number] : rule_number_map) {
    item_row_map[drc_rule] = ++row;
  }
  item_row_map["Total"] = ++row;

  irt_int column = 0;
  std::map<std::string, irt_int> item_column_map;
  for (auto& [source, drc_number_map] : source_number_map) {
    item_column_map[source] = ++column;
  }
  item_column_map["Total"] = ++column;

  // build table
  fort::char_table drc_table;
  drc_table.set_border_style(FT_SOLID_ROUND_STYLE);
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
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    irt_int column = item_column_map[GetVRSourceTypeName()(source)];
    for (auto& [drc_rule, row] : item_row_map) {
      if (RTUtil::exist(source_drc_number_map[source], drc_rule)) {
        drc_table[row][column] = RTUtil::getString(source_drc_number_map[source][drc_rule]);
      } else {
        drc_table[row][column] = "0";
      }
    }
  }
  // last row
  for (auto& [source, total_number] : source_number_map) {
    irt_int row = item_row_map["Total"];
    irt_int column = item_column_map[source];
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
  std::vector<std::vector<std::string>> table_list;
  table_list.push_back(RTUtil::splitString(wire_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(via_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(drc_table.to_string(), '\n'));
  int max_size = INT_MIN;
  for (std::vector<std::string>& table : table_list) {
    max_size = std::max(max_size, static_cast<int>(table.size()));
  }
  for (std::vector<std::string>& table : table_list) {
    for (irt_int i = table.size(); i < max_size; i++) {
      std::string table_str;
      table_str.append(table.front().length() / 3, ' ');
      table.push_back(table_str);
    }
  }

  for (irt_int i = 0; i < max_size; i++) {
    std::string table_str;
    for (std::vector<std::string>& table : table_list) {
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

}  // namespace irt
