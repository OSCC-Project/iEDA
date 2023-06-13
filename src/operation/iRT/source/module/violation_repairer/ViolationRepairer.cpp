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
  updateNetBlockageMap(vr_model);
}

void ViolationRepairer::updateNetBlockageMap(VRModel& vr_model)
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
  for (GridMap<VRGCell>& gcell_map : vr_model.get_layer_gcell_map()) {
    for (irt_int x = 0; x < gcell_map.get_x_size(); x++) {
      for (irt_int y = 0; y < gcell_map.get_y_size(); y++) {
        PlanarRect& gcell_rect = gcell_map[x][y].get_real_rect();
        for (auto& [net_idx, blockage_list] : gcell_map[x][y].get_net_blockage_map()) {
          for (PlanarRect& blockage : blockage_list) {
            if (RTUtil::isClosedOverlap(gcell_rect, blockage)) {
              continue;
            }
            LOG_INST.error(Loc::current(), "The region of gcell does not contain blockage!");
          }
        }
      }
    }
  }

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

#if 1  // repair ra_model

void ViolationRepairer::repairVRModel(VRModel& vr_model)
{
  Monitor monitor;

  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

  irt_int batch_size = RTUtil::getBatchSize(vr_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < vr_net_list.size(); i++) {
    repairVRNet(vr_model, vr_net_list[i]);
    if ((i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Processed ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  LOG_INST.info(Loc::current(), "Processed ", vr_net_list.size(), " nets", monitor.getStatsInfo());
}

void ViolationRepairer::repairVRNet(VRModel& vr_model, VRNet& vr_net)
{
  buildKeyCoordPinMap(vr_net);
  buildCoordTree(vr_net);
  buildPHYNodeResult(vr_net);
  repairMinArea(vr_net);
  updateNetBlockageMap(vr_model, vr_net);
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
  wire_node.set_first(first_coord);
  wire_node.set_second(second_coord);
  wire_node.set_wire_width(routing_layer_list[layer_idx].get_min_width());
  return (new TNode<PHYNode>(phy_node));
}

TNode<PHYNode>* ViolationRepairer::makeViaPHYNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _vr_data_manager.getDatabase().get_layer_via_master_list();

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
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();

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

void ViolationRepairer::repairMinArea(VRNet& vr_net)
{
}

void ViolationRepairer::updateNetBlockageMap(VRModel& vr_model, VRNet& vr_net)
{
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _vr_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<GridMap<VRGCell>>& layer_gcell_map = vr_model.get_layer_gcell_map();
  for (const LayerRect& real_rect : getRealRectList(vr_net.get_vr_result_tree())) {
    irt_int layer_idx = real_rect.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(real_rect);
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(real_rect, min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        layer_gcell_map[layer_idx][x][y].get_net_blockage_map()[vr_net.get_net_idx()].push_back(enlarged_real_rect);
      }
    }
  }
}

std::vector<LayerRect> ViolationRepairer::getRealRectList(MTree<PHYNode>& phy_node_tree)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _vr_data_manager.getDatabase().get_layer_via_master_list();

  std::vector<LayerRect> real_rect_list;
  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(phy_node_tree)) {
    PHYNode& phy_node = phy_node_node->value();
    if (phy_node.isType<WireNode>()) {
      WireNode& wire_node = phy_node.getNode<WireNode>();
      PlanarRect wire_rect = RTUtil::getEnlargedRect(wire_node.get_first(), wire_node.get_second(), wire_node.get_wire_width() / 2);
      real_rect_list.emplace_back(wire_rect, wire_node.get_layer_idx());
    } else if (phy_node.isType<ViaNode>()) {
      ViaNode& via_node = phy_node.getNode<ViaNode>();
      ViaMasterIdx& via_master_idx = via_node.get_via_master_idx();
      ViaMaster& via_master = layer_via_master_list[via_master_idx.get_below_layer_idx()][via_master_idx.get_via_idx()];
      for (const LayerRect& enclosure : {via_master.get_below_enclosure(), via_master.get_above_enclosure()}) {
        PlanarRect offset_enclosure = RTUtil::getOffsetRect(enclosure, via_node);
        real_rect_list.emplace_back(offset_enclosure, enclosure.get_layer_idx());
      }
    }
  }
  return real_rect_list;
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
  countVRModel(vr_model);
  reportTable(vr_model);
}

void ViolationRepairer::countVRModel(VRModel& vr_model)
{
  irt_int micron_dbu = _vr_data_manager.getDatabase().get_micron_dbu();
  GCellAxis& gcell_axis = _vr_data_manager.getDatabase().get_gcell_axis();

  VRModelStat& vr_model_stat = vr_model.get_vr_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = vr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_obs_rect_map
      = vr_model_stat.get_routing_net_and_obs_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_net_rect_map
      = vr_model_stat.get_routing_net_and_net_rect_map();

  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(vr_net.get_vr_result_tree())) {
      PHYNode& phy_node = phy_node_node->value();
      if (phy_node.isType<WireNode>()) {
        WireNode& wire_node = phy_node.getNode<WireNode>();
        double wire_length = RTUtil::getManhattanDistance(wire_node.get_first(), wire_node.get_second()) / 1.0 / micron_dbu;
        routing_wire_length_map[wire_node.get_layer_idx()] += wire_length;
      } else if (phy_node.isType<ViaNode>()) {
        ViaNode& via_node = phy_node.getNode<ViaNode>();
        cut_via_number_map[via_node.get_via_master_idx().get_below_layer_idx()]++;
      }
    }
  }
  std::set<irt_int> visited_net_idx_set;
  std::vector<GridMap<VRGCell>>& layer_gcell_map = vr_model.get_layer_gcell_map();
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    visited_net_idx_set.insert(vr_net.get_net_idx());

    for (LayerRect& real_rect : getRealRectList(vr_net.get_vr_result_tree())) {
      irt_int layer_idx = real_rect.get_layer_idx();
      GridMap<VRGCell>& gcell_map = layer_gcell_map[layer_idx];

      std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>> net_blockage_map;
      PlanarRect grid_rect = RTUtil::getClosedGridRect(real_rect, gcell_axis);
      for (irt_int x = grid_rect.get_lb_x(); x <= grid_rect.get_rt_x(); x++) {
        for (irt_int y = grid_rect.get_lb_y(); y <= grid_rect.get_rt_y(); y++) {
          for (auto& [net_idx, blockage_list] : gcell_map[x][y].get_net_blockage_map()) {
            if (vr_net.get_net_idx() == net_idx) {
              continue;
            }
            if (RTUtil::exist(visited_net_idx_set, net_idx)) {
              continue;
            }
            net_blockage_map[net_idx].insert(blockage_list.begin(), blockage_list.end());
          }
        }
      }
      for (auto& [net_idx, blockage_set] : net_blockage_map) {
        for (const PlanarRect& blockage : blockage_set) {
          if (RTUtil::isOpenOverlap(real_rect, blockage)) {
            PlanarRect violation_rect = RTUtil::getOverlap(real_rect, blockage);
            if (net_idx == -1) {
              routing_net_and_obs_rect_map[layer_idx].insert(violation_rect);
            } else {
              routing_net_and_net_rect_map[layer_idx].insert(violation_rect);
            }
          }
        }
      }
    }
  }
  double total_wire_length = 0;
  irt_int total_via_number = 0;
  irt_int total_net_and_obs_rect_number = 0;
  irt_int total_net_and_net_rect_number = 0;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    total_wire_length += wire_length;
  }
  for (auto& [cut_layer_idx, via_number] : cut_via_number_map) {
    total_via_number += via_number;
  }
  for (auto& [routing_layer_idx, rect_list] : routing_net_and_obs_rect_map) {
    total_net_and_obs_rect_number += static_cast<irt_int>(rect_list.size());
  }
  for (auto& [routing_layer_idx, rect_list] : routing_net_and_net_rect_map) {
    total_net_and_net_rect_number += static_cast<irt_int>(rect_list.size());
  }
  vr_model_stat.set_total_wire_length(total_wire_length);
  vr_model_stat.set_total_via_number(total_via_number);
  vr_model_stat.set_total_net_and_obs_rect_number(total_net_and_obs_rect_number);
  vr_model_stat.set_total_net_and_net_rect_number(total_net_and_net_rect_number);
}

void ViolationRepairer::reportTable(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _vr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _vr_data_manager.getDatabase().get_cut_layer_list();

  // wire table
  VRModelStat& vr_model_stat = vr_model.get_vr_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = vr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, irt_int>& cut_via_number_map = vr_model_stat.get_cut_via_number_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_obs_rect_map
      = vr_model_stat.get_routing_net_and_obs_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_net_rect_map
      = vr_model_stat.get_routing_net_and_net_rect_map();
  double total_wire_length = vr_model_stat.get_total_wire_length();
  irt_int total_via_number = vr_model_stat.get_total_via_number();
  irt_int total_net_and_obs_rect_number = vr_model_stat.get_total_net_and_obs_rect_number();
  irt_int total_net_and_net_rect_number = vr_model_stat.get_total_net_and_net_rect_number();

  // wire table
  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double wire_length = routing_wire_length_map[routing_layer.get_layer_idx()];
    wire_table << routing_layer.get_layer_name()
               << RTUtil::getString(wire_length, "(", RTUtil::getPercentage(wire_length, total_wire_length), "%)") << fort::endr;
  }
  wire_table << fort::header << "Total" << total_wire_length << fort::endr;
  // via table
  fort::char_table via_table;
  via_table.set_border_style(FT_SOLID_STYLE);
  via_table << fort::header << "Cut Layer"
            << "Via number" << fort::endr;
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int cut_via_number = cut_via_number_map[cut_layer.get_layer_idx()];
    via_table << cut_layer.get_layer_name()
              << RTUtil::getString(cut_via_number, "(", RTUtil::getPercentage(cut_via_number, total_via_number), "%)") << fort::endr;
  }
  via_table << fort::header << "Total" << total_via_number << fort::endr;

  // report wire via info
  std::vector<std::string> wire_str_list = RTUtil::splitString(wire_table.to_string(), '\n');
  std::vector<std::string> via_str_list = RTUtil::splitString(via_table.to_string(), '\n');
  for (size_t i = 0; i < std::max(wire_str_list.size(), via_str_list.size()); i++) {
    std::string table_str;
    if (i < wire_str_list.size()) {
      table_str += wire_str_list[i];
    }
    table_str += " ";
    if (i < via_str_list.size()) {
      table_str += via_str_list[i];
    }
    LOG_INST.info(Loc::current(), table_str);
  }

  // violation_table
  fort::char_table violation_table;
  violation_table.set_border_style(FT_SOLID_STYLE);
  violation_table << fort::header << "Routing Layer"
                  << "Net And Obs Rect Number"
                  << "Net And Net Rect Number" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int net_and_obs_rect_number = static_cast<irt_int>(routing_net_and_obs_rect_map[routing_layer.get_layer_idx()].size());
    irt_int net_and_net_rect_number = static_cast<irt_int>(routing_net_and_net_rect_map[routing_layer.get_layer_idx()].size());
    violation_table << routing_layer.get_layer_name()
                    << RTUtil::getString(net_and_obs_rect_number, "(",
                                         RTUtil::getPercentage(net_and_obs_rect_number, total_net_and_obs_rect_number), "%)")
                    << RTUtil::getString(net_and_net_rect_number, "(",
                                         RTUtil::getPercentage(net_and_net_rect_number, total_net_and_net_rect_number), "%)")
                    << fort::endr;
  }
  violation_table << fort::header << "Total" << total_net_and_obs_rect_number << total_net_and_net_rect_number << fort::endr;
  for (std::string table_str : RTUtil::splitString(violation_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
