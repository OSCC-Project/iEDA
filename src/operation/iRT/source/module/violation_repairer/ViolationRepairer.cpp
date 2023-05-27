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

void ViolationRepairer::initInst(Config& config, Database& database)
{
  if (_vr_instance == nullptr) {
    _vr_instance = new ViolationRepairer(config, database);
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

void ViolationRepairer::repair(std::vector<Net>& net_list)
{
  Monitor monitor;

  std::vector<VRNet> vr_net_list = _vr_data_manager.convertToVRNetList(net_list);
  repairVRNetList(vr_net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kViolationRepairer), " completed!", monitor.getStatsInfo());
}

// private

ViolationRepairer* ViolationRepairer::_vr_instance = nullptr;

void ViolationRepairer::init(Config& config, Database& database)
{
  _vr_data_manager.input(config, database);
}

void ViolationRepairer::repairVRNetList(std::vector<VRNet>& vr_net_list)
{
  VRModel vr_model = initVRModel(vr_net_list);
  buildVRModel(vr_model);
  checkVRModel(vr_model);
  repairVRModel(vr_model);
  updateVRModel(vr_model);
  reportVRModel(vr_model);
}

#if 1  // build vr_model

VRModel ViolationRepairer::initVRModel(std::vector<VRNet>& vr_net_list)
{
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  Die& die = _vr_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();

  VRModel vr_model;
  std::vector<GridMap<VRGCell>>& layer_gcell_map = vr_model.get_layer_gcell_map();
  layer_gcell_map.resize(routing_layer_list.size());
  for (size_t layer_idx = 0; layer_idx < layer_gcell_map.size(); layer_idx++) {
    GridMap<VRGCell>& gcell_map = layer_gcell_map[layer_idx];
    gcell_map.init(die.getXSize(), die.getYSize());
    for (irt_int x = 0; x < die.getXSize(); x++) {
      for (irt_int y = 0; y < die.getYSize(); y++) {
        VRGCell& vr_gcell = gcell_map[x][y];
        vr_gcell.set_real_rect(RTUtil::getRealRect(x, y, gcell_axis));
      }
    }
  }
  vr_model.set_vr_net_list(vr_net_list);

  return vr_model;
}

void ViolationRepairer::buildVRModel(VRModel& vr_model)
{
  addBlockageList(vr_model);
}

void ViolationRepairer::addBlockageList(VRModel& vr_model)
{
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _vr_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _vr_data_manager.getDatabase().get_routing_blockage_list();

  std::vector<GridMap<VRGCell>>& layer_gcell_map = vr_model.get_layer_gcell_map();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_blockage.get_real_rect());
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_blockage.get_real_rect(), min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        layer_gcell_map[layer_idx][x][y].get_net_blockage_map()[-1].push_back(enlarged_real_rect);
      }
    }
  }
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (const EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        irt_int layer_idx = routing_shape.get_layer_idx();
        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_shape.get_real_rect());
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_shape.get_real_rect(), min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            layer_gcell_map[layer_idx][x][y].get_net_blockage_map()[vr_net.get_net_idx()].push_back(enlarged_real_rect);
          }
        }
      }
    }
  }
}

#endif

#if 1  // check ra_model

void ViolationRepairer::checkVRModel(VRModel& vr_model)
{
}

#endif

#if 1  // repair ra_model

void ViolationRepairer::repairVRModel(VRModel& vr_model)
{
  buildVRResultTree(vr_model);
}

void ViolationRepairer::buildVRResultTree(VRModel& vr_model)
{
#pragma omp parallel for
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
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();

  if (RTUtil::isOblique(first_coord, second_coord)) {
    LOG_INST.error(Loc::current(), "The wire phy node is oblique!");
  }
  irt_int layer_idx = first_coord.get_layer_idx();
  if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
    LOG_INST.error(Loc::current(), "The wire layer_idx is illegal!");
  }
  RTUtil::sort(first_coord, second_coord, CmpLayerCoordByXASC());

  PHYNode phy_node;
  WireNode& wire_node = phy_node.getNode<WireNode>();
  wire_node.set_net_idx(vr_net.get_net_idx());
  wire_node.set_layer_idx(layer_idx);

  EXTPlanarCoord& wire_first_coord = wire_node.get_first();
  wire_first_coord.set_real_coord(first_coord);
  wire_first_coord.set_grid_coord(RTUtil::getGridCoord(wire_first_coord.get_real_coord(), gcell_axis, vr_net.get_bounding_box()));

  EXTPlanarCoord& wire_second_coord = wire_node.get_second();
  wire_second_coord.set_real_coord(second_coord);
  wire_second_coord.set_grid_coord(RTUtil::getGridCoord(wire_second_coord.get_real_coord(), gcell_axis, vr_net.get_bounding_box()));

  wire_node.set_wire_width(routing_layer_list[layer_idx].get_min_width());
  return (new TNode<PHYNode>(phy_node));
}

TNode<PHYNode>* ViolationRepairer::makeViaPHYNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord)
{
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _vr_data_manager.getDatabase().get_layer_via_master_list();

  if (below_layer_idx < 0 || below_layer_idx >= static_cast<irt_int>(layer_via_master_list.size())) {
    LOG_INST.error(Loc::current(), "The via below_layer_idx is illegal!");
  }
  PHYNode phy_node;
  ViaNode& via_node = phy_node.getNode<ViaNode>();
  via_node.set_net_idx(vr_net.get_net_idx());
  via_node.set_real_coord(coord);
  via_node.set_grid_coord(RTUtil::getGridCoord(via_node.get_real_coord(), gcell_axis, vr_net.get_bounding_box()));
  via_node.set_via_idx(layer_via_master_list[below_layer_idx].front().get_via_idx());
  return (new TNode<PHYNode>(phy_node));
}

TNode<PHYNode>* ViolationRepairer::makePinPHYNode(VRNet& vr_net, irt_int pin_idx, LayerCoord coord)
{
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();

  irt_int layer_idx = coord.get_layer_idx();
  if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
    LOG_INST.error(Loc::current(), "The pin layer_idx is illegal!");
  }
  PHYNode phy_node;
  PinNode& pin_node = phy_node.getNode<PinNode>();
  pin_node.set_net_idx(vr_net.get_net_idx());
  pin_node.set_pin_idx(pin_idx);
  pin_node.set_real_coord(coord);
  pin_node.set_grid_coord(RTUtil::getGridCoord(pin_node.get_real_coord(), gcell_axis, vr_net.get_bounding_box()));
  pin_node.set_layer_idx(layer_idx);
  return (new TNode<PHYNode>(phy_node));
}

#endif

#if 1  // plot ra_model

void ViolationRepairer::plotVRModel(VRModel& vr_model)
{
}

#endif

#if 1  // update ra_model

void ViolationRepairer::updateVRModel(VRModel& vr_model)
{
  updateOriginVRResultTree(vr_model);
}

void ViolationRepairer::updateOriginVRResultTree(VRModel& vr_model)
{
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    Net* origin_net = vr_net.get_origin_net();
    origin_net->set_vr_result_tree(vr_net.get_vr_result_tree());
  }
}

#endif

#if 1  // report ra_model

void ViolationRepairer::reportVRModel(VRModel& vr_model)
{
}

#endif

}  // namespace irt
