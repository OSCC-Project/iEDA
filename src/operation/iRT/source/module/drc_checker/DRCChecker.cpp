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
#include "DRCChecker.hpp"

#include "DRCRect.hpp"
#include "GDSPlotter.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void DRCChecker::initInst()
{
  if (_dc_instance == nullptr) {
    _dc_instance = new DRCChecker();
  }
}

DRCChecker& DRCChecker::getInst()
{
  if (_dc_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_dc_instance;
}

void DRCChecker::destroyInst()
{
  if (_dc_instance != nullptr) {
    delete _dc_instance;
    _dc_instance = nullptr;
  }
}

// function

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<DRCRect> drc_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    for (DRCRect& drc_rect : getDRCRectList(net_idx, segment)) {
      drc_rect_list.push_back(drc_rect);
    }
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, Segment<LayerCoord>& segment)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  irt_int first_layer_idx = first_coord.get_layer_idx();
  irt_int second_layer_idx = second_coord.get_layer_idx();
  if (first_layer_idx != second_layer_idx) {
    RTUtil::swapASC(first_layer_idx, second_layer_idx);
    for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
      ViaMaster& via_master = layer_via_master_list[layer_idx].front();

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
      drc_rect_list.emplace_back(net_idx, offset_above_enclosure, true);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
      drc_rect_list.emplace_back(net_idx, offset_below_enclosure, true);

      for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
        LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
        drc_rect_list.emplace_back(net_idx, offset_cut_shape, false);
      }
    }
  } else {
    irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
    LayerRect wire_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
    drc_rect_list.emplace_back(net_idx, wire_rect, true);
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, MTree<LayerCoord>& coord_tree)
{
  std::vector<DRCRect> drc_rect_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(coord_tree)) {
    LayerCoord first_coord = coord_segment.get_first()->value();
    LayerCoord second_coord = coord_segment.get_second()->value();
    Segment<LayerCoord> segment(first_coord, second_coord);
    for (DRCRect& drc_rect : getDRCRectList(net_idx, segment)) {
      drc_rect_list.push_back(drc_rect);
    }
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, MTree<PHYNode>& phy_node_tree)
{
  std::vector<DRCRect> drc_rect_list;
  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(phy_node_tree)) {
    for (DRCRect& drc_rect : getDRCRectList(net_idx, phy_node_node->value())) {
      drc_rect_list.push_back(drc_rect);
    }
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, PHYNode& phy_node)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  if (phy_node.isType<WireNode>()) {
    WireNode& wire_node = phy_node.getNode<WireNode>();
    PlanarRect wire_rect = RTUtil::getEnlargedRect(wire_node.get_first(), wire_node.get_second(), wire_node.get_wire_width() / 2);
    drc_rect_list.emplace_back(net_idx, LayerRect(wire_rect, wire_node.get_layer_idx()), true);
  } else if (phy_node.isType<ViaNode>()) {
    ViaNode& via_node = phy_node.getNode<ViaNode>();
    ViaMasterIdx& via_master_idx = via_node.get_via_master_idx();
    ViaMaster& via_master = layer_via_master_list[via_master_idx.get_below_layer_idx()][via_master_idx.get_via_idx()];

    LayerRect& above_enclosure = via_master.get_above_enclosure();
    LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, via_node), above_enclosure.get_layer_idx());
    drc_rect_list.emplace_back(net_idx, offset_above_enclosure, true);

    LayerRect& below_enclosure = via_master.get_below_enclosure();
    LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, via_node), below_enclosure.get_layer_idx());
    drc_rect_list.emplace_back(net_idx, offset_below_enclosure, true);

    for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
      LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, via_node), via_master.get_cut_layer_idx());
      drc_rect_list.emplace_back(net_idx, offset_cut_shape, false);
    }
  }
  return drc_rect_list;
}

RegionQuery* DRCChecker::initRegionQuery()
{
  RegionQuery* region_query = new RegionQuery();
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    region_query->set_idrc_region_query(RTAPI_INST.initRegionQuery());
  }
  return region_query;
}

void DRCChecker::destoryRegionQuery(RegionQuery* region_query)
{
  if (region_query != nullptr) {
    void* idrc_region_query = region_query->get_idrc_region_query();
    if (idrc_region_query != nullptr) {
      RTAPI_INST.destroyRegionQuery(idrc_region_query);
      idrc_region_query = nullptr;
    }

    region_query->get_routing_net_rect_map().clear();
    region_query->get_cut_net_rect_map().clear();

    auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
    auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
    for (auto& [net_id, layer_shape_map] : routing_net_shape_map) {
      for (auto& [layer_idx, shape_map] : layer_shape_map) {
        for (auto& [rect, shape_ptr] : shape_map) {
          if (shape_ptr != nullptr) {
            delete shape_ptr;
            shape_ptr = nullptr;
          }
        }
      }
    }
    for (auto& [net_id, layer_shape_map] : cut_net_shape_map) {
      for (auto& [layer_idx, shape_map] : layer_shape_map) {
        for (auto& [rect, shape_ptr] : shape_map) {
          if (shape_ptr != nullptr) {
            delete shape_ptr;
            shape_ptr = nullptr;
          }
        }
      }
    }
    routing_net_shape_map.clear();
    cut_net_shape_map.clear();

    region_query->get_routing_region_map().clear();
    region_query->get_cut_region_map().clear();

    delete region_query;
    region_query = nullptr;
  }
}

std::map<irt_int, std::map<irt_int, std::set<LayerRect, CmpLayerRectByXASC>>>& DRCChecker::getLayerNetRectMap(RegionQuery* region_query,
                                                                                                              bool is_routing)
{
  if (is_routing) {
    return region_query->get_routing_net_rect_map();
  } else {
    return region_query->get_cut_net_rect_map();
  }
}

void DRCChecker::addEnvRectList(RegionQuery* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  addEnvRectList(region_query, drc_rect_list);
}

void DRCChecker::addEnvRectList(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  addNetRectMap(region_query, drc_rect_list);
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    RTAPI_INST.addEnvRectList(region_query, ids_rect_list);
  } else {
    addEnvRectListByRTDRC(region_query, drc_rect_list);
  }
}

void DRCChecker::delEnvRectList(RegionQuery* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  delEnvRectList(region_query, drc_rect_list);
}

void DRCChecker::delEnvRectList(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  delNetRectMap(region_query, drc_rect_list);
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    RTAPI_INST.delEnvRectList(region_query->get_idrc_region_query(), ids_rect_list);
  } else {
    delEnvRectListByRTDRC(region_query, drc_rect_list);
  }
}

bool DRCChecker::hasViolation(RegionQuery* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return hasViolation(region_query, drc_rect_list);
}

bool DRCChecker::hasViolation(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  for (auto [drc, num] : getViolation(region_query, drc_rect_list)) {
    if (num > 0) {
      return true;
    }
  }
  return false;
}

bool DRCChecker::hasViolation(RegionQuery* region_query)
{
  for (auto [drc, num] : getViolationByRTDRC(region_query)) {
    if (num > 0) {
      return true;
    }
  }
  return false;
}

bool DRCChecker::hasViolation(const std::vector<DRCRect>& drc_rect_list)
{
  for (auto [drc, num] : getViolation(drc_rect_list)) {
    if (num > 0) {
      return true;
    }
  }
  return false;
}

std::map<std::string, int> DRCChecker::getViolation(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  std::map<std::string, irt_int> violation_name_num_map;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    violation_name_num_map = RTAPI_INST.getViolation(region_query->get_idrc_region_query(), ids_rect_list);
  } else {
    violation_name_num_map = getViolationByRTDRC(region_query, drc_rect_list);
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolation(RegionQuery* region_query)
{
  std::map<std::string, irt_int> violation_name_num_map;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    violation_name_num_map = RTAPI_INST.getViolation(region_query->get_idrc_region_query());
  } else {
    violation_name_num_map = getViolationByRTDRC(region_query);
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolation(const std::vector<DRCRect>& drc_rect_list)
{
  std::map<std::string, irt_int> violation_name_num_map;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    // violation_name_num_map = RTAPI_INST.getViolation(region_query->get_idrc_region_query());
  } else {
    violation_name_num_map = getViolationByRTDRC(drc_rect_list);
  }
  return violation_name_num_map;
}

std::vector<LayerRect> DRCChecker::getMaxScope(const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  std::vector<LayerRect> max_scope_list;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    max_scope_list = RTAPI_INST.getMaxScope(ids_rect_list);
  } else {
    max_scope_list = getMinSpacingRect(ids_rect_list);
  }
  return max_scope_list;
}

std::vector<LayerRect> DRCChecker::getMinScope(const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  std::vector<LayerRect> min_scope_list;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    min_scope_list = RTAPI_INST.getMinScope(ids_rect_list);
  } else {
    min_scope_list = getMinSpacingRect(ids_rect_list);
  }
  return min_scope_list;
}

std::vector<LayerRect> DRCChecker::getMaxScope(const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return getMaxScope(drc_rect_list);
}

std::vector<LayerRect> DRCChecker::getMinScope(const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return getMinScope(drc_rect_list);
}

void DRCChecker::plotRegionQuery(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    return;
  } else {
    RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
    std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);
    plotRegionQueryByRTDRC(rt_region_query, ids_rect_list);
  }
}

#if 1  // violation info

std::vector<ViolationInfo> DRCChecker::getViolationInfo(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ViolationInfo> violation_info_list;
  checkMinSpacingByOther(region_query, drc_rect_list, violation_info_list);
  // checkMinSpacingBySelf(drc_rect_list, violation_info_list);
  // checkMinArea();
  return violation_info_list;
}
std::vector<ViolationInfo> DRCChecker::getViolationInfo(RegionQuery* region_query)
{
  std::vector<ViolationInfo> violation_info_list;
  checkMinSpacingBySelf(region_query, violation_info_list);
  return violation_info_list;
}
std::vector<ViolationInfo> DRCChecker::getViolationInfo(const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ViolationInfo> violation_info_list;
  checkMinSpacingBySelf(drc_rect_list, violation_info_list);
  return violation_info_list;
}

#endif

// private

DRCChecker* DRCChecker::_dc_instance = nullptr;

// function
std::vector<ids::DRCRect> DRCChecker::convertToIDSRect(const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list;
  for (DRCRect drc_rect : drc_rect_list) {
    ids_rect_list.push_back(RTAPI_INST.convertToIDSRect(drc_rect.get_net_idx(), drc_rect.get_layer_rect(), drc_rect.get_is_routing()));
  }
  return ids_rect_list;
}

void DRCChecker::addNetRectMap(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  for (const DRCRect& drc_rect : drc_rect_list) {
    irt_int net_id = drc_rect.get_net_idx();
    const LayerRect& rect = drc_rect.get_layer_rect();
    irt_int layer_idx = rect.get_layer_idx();
    if (drc_rect.get_is_routing()) {
      region_query->get_routing_net_rect_map()[layer_idx][net_id].insert(rect);
    } else {
      region_query->get_cut_net_rect_map()[layer_idx][net_id].insert(rect);
    }
  }
}

void DRCChecker::addEnvRectListByRTDRC(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCRect& drc_rect : drc_rect_list) {
    const LayerRect& layer_rect = drc_rect.get_layer_rect();
    BoostBox shape = RTUtil::convertToBoostBox(layer_rect);
    irt_int layer_idx = layer_rect.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(layer_rect);
    BoostBox enlarged_shape = RTUtil::enlargeBoostBox(shape, min_spacing);

    RQShape* rq_shape = new RQShape();
    rq_shape->set_shape(shape);
    rq_shape->set_net_id(drc_rect.get_net_idx());
    rq_shape->set_routing_layer_idx(layer_idx);
    rq_shape->set_min_spacing(min_spacing);
    rq_shape->set_enlarged_shape(enlarged_shape);

    if (drc_rect.get_is_routing()) {
      routing_net_shape_map[rq_shape->get_net_id()][layer_idx][layer_rect] = rq_shape;
      routing_region_map[layer_idx].insert(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape));
    } else {
      cut_net_shape_map[rq_shape->get_net_id()][layer_idx][layer_rect] = rq_shape;
      cut_region_map[layer_idx].insert(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape));
    }
  }
}

void DRCChecker::delNetRectMap(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  for (const DRCRect& drc_rect : drc_rect_list) {
    irt_int net_id = drc_rect.get_net_idx();
    const LayerRect& rect = drc_rect.get_layer_rect();
    irt_int layer_idx = rect.get_layer_idx();

    auto& net_rect_map = drc_rect.get_is_routing() ? region_query->get_routing_net_rect_map() : region_query->get_cut_net_rect_map();

    if (!RTUtil::exist(net_rect_map, layer_idx) || !RTUtil::exist(net_rect_map[layer_idx], net_id)
        || !RTUtil::exist(net_rect_map[layer_idx][net_id], rect)) {
      LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
    }
    net_rect_map[layer_idx][net_id].erase(rect);
  }
}

void DRCChecker::delEnvRectListByRTDRC(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCRect& drc_rect : drc_rect_list) {
    irt_int net_id = drc_rect.get_net_idx();
    const LayerRect layer_rect = drc_rect.get_layer_rect();
    irt_int layer_idx = layer_rect.get_layer_idx();

    RQShape* rq_shape = nullptr;
    if (drc_rect.get_is_routing()) {
      // 从obj map中删除数据
      if (!RTUtil::exist(routing_net_shape_map, net_id) || !RTUtil::exist(routing_net_shape_map[net_id], layer_idx)
          || !RTUtil::exist(routing_net_shape_map[net_id][layer_idx], layer_rect)) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
      RQShape* rq_shape = routing_net_shape_map[net_id][layer_idx][layer_rect];

      routing_net_shape_map[net_id][layer_idx].erase(layer_rect);
      // 从rtree中删除数据
      if (!RTUtil::exist(routing_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
      if (!routing_region_map[layer_idx].remove(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape))) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
    } else {
      // 从obj map中删除数据
      if (!RTUtil::exist(cut_net_shape_map, net_id) || !RTUtil::exist(cut_net_shape_map[net_id], layer_idx)
          || !RTUtil::exist(cut_net_shape_map[net_id][layer_idx], layer_rect)) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
      RQShape* rq_shape = cut_net_shape_map[net_id][layer_idx][layer_rect];

      cut_net_shape_map[net_id][layer_idx].erase(layer_rect);
      // 从rtree中删除数据
      if (!RTUtil::exist(cut_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
      if (!cut_region_map[layer_idx].remove(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape))) {
        LOG_INST.error(Loc::current(), "There is no rect net : ", net_id, " layer idx :", layer_idx, "!");
      }
    }
    // 释放资源
    delete rq_shape;
    rq_shape = nullptr;
  }
}

std::map<std::string, int> DRCChecker::getViolationByRTDRC(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::map<std::string, irt_int> violation_name_num_map;
  violation_name_num_map.insert(std::make_pair("Cut EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Enclosure", 0));
  violation_name_num_map.insert(std::make_pair("Metal EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Short", 0));
  violation_name_num_map.insert(std::make_pair("Metal Parallel Run Length Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Notch Spacing", 0));
  violation_name_num_map.insert(std::make_pair("MinStep", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Area", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Corner Fill Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Hole Area", 0));

  std::map<irt_int, std::vector<RQShape>> net_shape_list_map;
  for (const DRCRect& drc_rect : drc_rect_list) {
    RQShape rq_shape = convertToRQShape(drc_rect);
    net_shape_list_map[rq_shape.get_net_id()].push_back(rq_shape);
  }

  for (auto& [net_id, shape_list] : net_shape_list_map) {
    // check drc by other
    for (auto [violation_name, num] : checkByOtherByRTDRC(region_query, shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolationByRTDRC(RegionQuery* region_query)
{
  std::map<std::string, irt_int> violation_name_num_map;
  violation_name_num_map.insert(std::make_pair("Cut EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Enclosure", 0));
  violation_name_num_map.insert(std::make_pair("Metal EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Short", 0));
  violation_name_num_map.insert(std::make_pair("Metal Parallel Run Length Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Notch Spacing", 0));
  violation_name_num_map.insert(std::make_pair("MinStep", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Area", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Corner Fill Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Hole Area", 0));

  auto& routing_net_rect_map = region_query->get_routing_net_shape_map();

  for (auto& [net_id, layer_shape_list] : routing_net_rect_map) {
    std::vector<RQShape> rq_shape_list;
    for (auto& [layer_idx, shape_map] : layer_shape_list) {
      for (auto& [real_rect, rq_shape] : shape_map) {
        rq_shape_list.push_back(*rq_shape);
      }
    }
    // check drc by self
    for (auto [violation_name, num] : checkBySelfByRTDRC(rq_shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolationByRTDRC(const std::vector<DRCRect>& drc_shape_list)
{
  std::map<std::string, irt_int> violation_name_num_map;
  violation_name_num_map.insert(std::make_pair("Cut EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Enclosure", 0));
  violation_name_num_map.insert(std::make_pair("Metal EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Short", 0));
  violation_name_num_map.insert(std::make_pair("Metal Parallel Run Length Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Notch Spacing", 0));
  violation_name_num_map.insert(std::make_pair("MinStep", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Area", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Corner Fill Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Hole Area", 0));

  std::map<irt_int, std::vector<RQShape>> net_shape_list_map;
  for (const DRCRect& drc_shape : drc_shape_list) {
    RQShape rq_shape = convertToRQShape(drc_shape);
    net_shape_list_map[rq_shape.get_net_id()].push_back(rq_shape);
  }

  for (auto& [net_id, shape_list] : net_shape_list_map) {
    for (auto [violation_name, num] : checkBySelfByRTDRC(shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
  }
  return violation_name_num_map;
}

RQShape DRCChecker::convertToRQShape(const DRCRect& drc_rect)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  const LayerRect& layer_rect = drc_rect.get_layer_rect();
  BoostBox shape = RTUtil::convertToBoostBox(layer_rect);
  irt_int layer_idx = layer_rect.get_layer_idx();
  irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(layer_rect);
  BoostBox enlarged_shape = RTUtil::enlargeBoostBox(shape, min_spacing);

  RQShape rq_shape;
  rq_shape.set_net_id(drc_rect.get_net_idx());
  rq_shape.set_shape(shape);
  rq_shape.set_is_routing(drc_rect.get_is_routing());
  rq_shape.set_routing_layer_idx(layer_idx);
  rq_shape.set_min_spacing(min_spacing);
  rq_shape.set_enlarged_shape(enlarged_shape);
  return rq_shape;
}

std::map<std::string, int> DRCChecker::checkByOtherByRTDRC(RegionQuery* region_query, std::vector<RQShape>& drc_shape_list)
{
  auto& routing_region_map = region_query->get_routing_region_map();

  std::map<std::string, int> violation_name_num_map;
  for (RQShape& drc_shape : drc_shape_list) {
    bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = routing_region_map[drc_shape.get_routing_layer_idx()];

    // 查询重叠
    std::vector<std::pair<BoostBox, RQShape*>> result_list;
    rtree.query(bgi::intersects(drc_shape.get_enlarged_shape()), std::back_inserter(result_list));

    // 遍历每个重叠 判断是否Spacing违例
    for (size_t i = 0; i < result_list.size(); i++) {
      RQShape* overlap_shape = result_list[i].second;
      if (overlap_shape->get_net_id() == drc_shape.get_net_id()) {
        continue;
      }
      irt_int require_spacing = std::max(overlap_shape->get_min_spacing(), drc_shape.get_min_spacing());
      irt_int spacing = RTUtil::getEuclideanDistance(overlap_shape->get_shape(), drc_shape.get_shape());
      if (spacing < require_spacing) {
        violation_name_num_map["RT Spacing"]++;
      }
    }
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::checkBySelfByRTDRC(std::vector<RQShape>& drc_shape_list)
{
  std::map<std::string, int> violation_name_num_map;

  std::map<irt_int, std::vector<RQShape>> layer_net_shape_list_map;
  for (RQShape& drc_shape : drc_shape_list) {
    layer_net_shape_list_map[drc_shape.get_routing_layer_idx()].push_back(drc_shape);
  }

  for (auto& [layer_idx, net_shape_list] : layer_net_shape_list_map) {
    for (size_t i = 0; i < net_shape_list.size(); i++) {
      for (size_t j = i + 1; j < net_shape_list.size(); j++) {
        RQShape& net_shape1 = net_shape_list[i];
        RQShape& net_shape2 = net_shape_list[j];
        if (RTUtil::isOverlap(net_shape1.get_shape(), net_shape2.get_shape())) {
          continue;
        }
        if (checkMinSpacingByRTDRC(net_shape1, net_shape2, net_shape_list)) {
          continue;
        }
        violation_name_num_map["RT Self net"]++;
      }
    }
  }
  return violation_name_num_map;
}

bool DRCChecker::checkMinSpacingByRTDRC(RQShape& net_shape1, RQShape& net_shape2, std::vector<RQShape>& net_shape_list)
{
  irt_int require_spacing = std::max(net_shape1.get_min_spacing(), net_shape2.get_min_spacing());
  irt_int spacing = RTUtil::getEuclideanDistance(net_shape1.get_shape(), net_shape2.get_shape());

  if (spacing < require_spacing) {
    BoostBox& box1 = net_shape1.get_shape();
    BoostBox& box2 = net_shape2.get_shape();

    std::vector<irt_int> x_list = {box1.min_corner().x(), box1.max_corner().x(), box2.min_corner().x(), box2.max_corner().x()};
    std::vector<irt_int> y_list = {box1.min_corner().y(), box1.max_corner().y(), box2.min_corner().y(), box2.max_corner().y()};

    std::sort(x_list.begin(), x_list.end());
    std::sort(y_list.begin(), y_list.end());

    BoostBox violation_region(BoostPoint(x_list[1], y_list[1]), BoostPoint(x_list[2], y_list[2]));

    for (size_t i = 0; i < net_shape_list.size(); i++) {
      if (bg::covered_by(violation_region, net_shape_list[i].get_shape())) {
        return true;
      }
    }
    return false;
  }
  return true;
}

std::vector<LayerRect> DRCChecker::getMinSpacingRect(const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = DM_INST.getHelper().get_routing_layer_name_to_idx_map();

  std::vector<LayerRect> min_scope_list;
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    if (!RTUtil::exist(routing_layer_name_to_idx_map, drc_rect.layer_name)) {
      continue;
    }
    PlanarRect rect(drc_rect.lb_x, drc_rect.lb_y, drc_rect.rt_x, drc_rect.rt_y);
    irt_int layer_idx = DM_INST.getHelper().getRoutingLayerIdxByName(drc_rect.layer_name);
    RoutingLayer& routing_layer = DM_INST.getDatabase().get_routing_layer_list()[layer_idx];
    min_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, routing_layer.getMinSpacing(rect)), layer_idx);
  }
  return min_scope_list;
}

void DRCChecker::plotRegionQueryByRTDRC(RegionQuery* region_query, const std::vector<ids::DRCRect>& drc_rect_list)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string gp_temp_directory_path = DM_INST.getConfig().gp_temp_directory_path;

  auto& routing_net_rect_map = region_query->get_routing_net_shape_map();

  GPGDS gp_gds;

  // base_region
  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(die.get_real_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  // scale_axis
  GPStruct box_track_axis_struct("scale_axis");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
      for (irt_int x = x_grid.get_start_line(); x <= x_grid.get_end_line(); x += x_grid.get_step_length()) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
        gp_path.set_segment(x, die.get_real_lb_y(), x, die.get_real_rt_y());
        gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        box_track_axis_struct.push(gp_path);
      }
    }
    for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
      for (irt_int y = y_grid.get_start_line(); y <= y_grid.get_end_line(); y += y_grid.get_step_length()) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
        gp_path.set_segment(die.get_real_lb_x(), y, die.get_real_rt_x(), y);
        gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        box_track_axis_struct.push(gp_path);
      }
    }
  }
  gp_gds.addStruct(box_track_axis_struct);

  // env shape
  for (auto& [net_id, layer_shape_list] : routing_net_rect_map) {
    GPStruct net_shape_struct(RTUtil::getString("env shape(net_", net_id, ")"));
    GPGraphType type = net_id == -1 ? GPGraphType::kBlockAndPin : GPGraphType::kKnownPanel;
    for (auto& [layer_idx, shape_map] : layer_shape_list) {
      for (auto& [rect, shape] : shape_map) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<irt_int>(type));
        gp_boundary.set_rect(RTUtil::convertToPlanarRect(shape->get_shape()));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
        net_shape_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(net_shape_struct);
  }

  // check shape
  GPStruct check_shape_struct(RTUtil::getString("check shape"));
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
    gp_boundary.set_rect(PlanarRect(drc_rect.lb_x, drc_rect.lb_y, drc_rect.rt_x, drc_rect.rt_y));
    gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(DM_INST.getHelper().getRoutingLayerIdxByName(drc_rect.layer_name)));
    check_shape_struct.push(gp_boundary);
  }
  gp_gds.addStruct(check_shape_struct);

  std::string gds_file_path = RTUtil::getString(gp_temp_directory_path, "region_query_.gds");
  GP_INST.plot(gp_gds, gds_file_path, false, false);
}

#if 1  // violation info

void DRCChecker::checkMinSpacingByOther(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list,
                                        std::vector<ViolationInfo>& violation_info_list)
{
  for (const DRCRect& drc_rect : drc_rect_list) {
    RQShape drc_shape = convertToRQShape(drc_rect);
    // 查询重叠
    std::vector<std::pair<BoostBox, RQShape*>> result_list;
    if (drc_shape.get_is_routing()) {
      auto& routing_region_map = region_query->get_routing_region_map();
      bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = routing_region_map[drc_shape.get_routing_layer_idx()];
      rtree.query(bgi::intersects(drc_shape.get_enlarged_shape()), std::back_inserter(result_list));
    } else {
      auto& cut_region_map = region_query->get_cut_region_map();
      bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = cut_region_map[drc_shape.get_routing_layer_idx()];
      rtree.query(bgi::intersects(drc_shape.get_enlarged_shape()), std::back_inserter(result_list));
    }

    // 遍历每个重叠 判断是否Spacing违例
    for (size_t i = 0; i < result_list.size(); i++) {
      RQShape* overlap_shape = result_list[i].second;
      if (overlap_shape->get_net_id() == drc_shape.get_net_id()) {
        continue;
      }
      irt_int require_spacing = std::max(overlap_shape->get_min_spacing(), drc_shape.get_min_spacing());
      irt_int spacing = RTUtil::getEuclideanDistance(overlap_shape->get_shape(), drc_shape.get_shape());
      if (spacing >= require_spacing) {
        continue;
      }

      PlanarRect check_rect1 = RTUtil::convertToPlanarRect(drc_shape.get_shape());
      PlanarRect check_rect2 = RTUtil::convertToPlanarRect(overlap_shape->get_shape());
      PlanarRect spacing_rect = RTUtil::getEnlargedRect(check_rect1, require_spacing);
      if (!RTUtil::isOverlap(spacing_rect, check_rect2)) {
        LOG_INST.error(Loc::current(), "Spacing violation rect is not overlap!");
      }
      LayerRect violation_region(RTUtil::getOverlap(spacing_rect, check_rect2), drc_shape.get_routing_layer_idx());

      std::map<irt_int, std::vector<irt::LayerRect>> violation_net_shape_map;
      violation_net_shape_map[drc_shape.get_net_id()].push_back(check_rect1);
      violation_net_shape_map[overlap_shape->get_net_id()].push_back(check_rect2);

      ViolationInfo violation;
      violation.set_is_routing(drc_rect.get_is_routing());
      violation.set_rule_name("RT Spacing");
      violation.set_violation_region(violation_region);
      violation.set_net_shape_map(violation_net_shape_map);
      violation_info_list.push_back(violation);
    }
  }
}

void DRCChecker::checkMinSpacingBySelf(RegionQuery* region_query, std::vector<ViolationInfo>& violation_info_list)
{
  std::map<irt_int, std::vector<RQShape>> net_shape_map;
  for (auto& [net_id, layer_shape_list] : region_query->get_routing_net_shape_map()) {
    std::vector<RQShape> rq_shape_list;
    for (auto& [layer_idx, shape_map] : layer_shape_list) {
      for (auto& [real_rect, rq_shape] : shape_map) {
        rq_shape_list.push_back(*rq_shape);
      }
    }
    net_shape_map[net_id] = rq_shape_list;
  }
  checkMinSpacingBySelf(net_shape_map, violation_info_list);
}

void DRCChecker::checkMinSpacingBySelf(const std::vector<DRCRect>& drc_rect_list, std::vector<ViolationInfo>& violation_info_list)
{
  std::map<irt_int, std::vector<RQShape>> net_shape_map;
  for (const DRCRect& drc_rect : drc_rect_list) {
    RQShape rq_shape = convertToRQShape(drc_rect);
    net_shape_map[rq_shape.get_net_id()].push_back(rq_shape);
  }
  checkMinSpacingBySelf(net_shape_map, violation_info_list);
}

void DRCChecker::checkMinSpacingBySelf(std::map<irt_int, std::vector<RQShape>>& net_shape_map,
                                       std::vector<ViolationInfo>& violation_info_list)
{
  for (auto& [net_id, shape_list] : net_shape_map) {
    std::map<irt_int, std::vector<RQShape>> layer_shape_map;
    for (RQShape& drc_shape : shape_list) {
      layer_shape_map[drc_shape.get_routing_layer_idx()].push_back(drc_shape);
    }

    for (auto& [layer_idx, net_shape_list] : layer_shape_map) {
      for (size_t i = 0; i < net_shape_list.size(); i++) {
        for (size_t j = i + 1; j < net_shape_list.size(); j++) {
          RQShape& net_shape1 = net_shape_list[i];
          RQShape& net_shape2 = net_shape_list[j];
          if (RTUtil::isOverlap(net_shape1.get_shape(), net_shape2.get_shape())) {
            continue;
          }
          if (checkMinSpacingByRTDRC(net_shape1, net_shape2, net_shape_list)) {
            continue;
          }

          irt_int max_spacing = std::max(net_shape1.get_min_spacing(), net_shape2.get_min_spacing());
          PlanarRect check_rect1 = RTUtil::convertToPlanarRect(net_shape1.get_shape());
          PlanarRect check_rect2 = RTUtil::convertToPlanarRect(net_shape2.get_shape());
          PlanarRect spacing_rect = RTUtil::getEnlargedRect(check_rect1, max_spacing);
          if (!RTUtil::isOverlap(spacing_rect, check_rect2)) {
            LOG_INST.error(Loc::current(), "Spacing violation rect is not overlap!");
          }
          LayerRect violation_region(RTUtil::getOverlap(spacing_rect, check_rect2), layer_idx);

          std::map<irt_int, std::vector<LayerRect>> violation_net_shape_map;
          violation_net_shape_map[net_id].emplace_back(LayerRect(check_rect1, layer_idx));
          violation_net_shape_map[net_id].emplace_back(LayerRect(check_rect2, layer_idx));

          ViolationInfo violation;
          violation.set_is_routing(net_shape1.get_is_routing());
          violation.set_rule_name("RT Self net");
          violation.set_violation_region(violation_region);
          violation.set_net_shape_map(violation_net_shape_map);
          violation_info_list.push_back(violation);
        }
      }
    }
  }
}

#endif

}  // namespace irt
