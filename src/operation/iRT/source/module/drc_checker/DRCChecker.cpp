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

#if 1  // 获得DRCRectList

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
  } else if (phy_node.isType<PatchNode>()) {
    PatchNode& patch_node = phy_node.getNode<PatchNode>();
    drc_rect_list.emplace_back(net_idx, patch_node, true);
  }
  return drc_rect_list;
}

#endif

#if 1  // 返回RegionQuery中的形状map

std::map<irt_int, std::map<irt_int, std::set<LayerRect, CmpLayerRectByXASC>>>& DRCChecker::getLayerNetRectMap(RegionQuery& region_query,
                                                                                                              bool is_routing)
{
  if (is_routing) {
    return region_query.get_routing_net_rect_map();
  } else {
    return region_query.get_cut_net_rect_map();
  }
}

#endif

#if 1  // 更新RegionQuery的RectList

void DRCChecker::updateRectList(RegionQuery& region_query, ChangeType change_type, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  updateRectList(region_query, change_type, drc_rect_list);
}

void DRCChecker::updateRectList(RegionQuery& region_query, ChangeType change_type, const std::vector<DRCRect>& drc_rect_list)
{
  RegionQuery* region_query_ref = &region_query;

  if (change_type == ChangeType::kAdd) {
    addEnvRectList(region_query_ref, drc_rect_list);
  } else if (change_type == ChangeType::kDel) {
    delEnvRectList(region_query_ref, drc_rect_list);
  }
}

#endif

#if 1  // 碰撞一定会产生DRC的最小膨胀矩形

std::vector<LayerRect> DRCChecker::getMinScope(const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return getMinScope(drc_rect_list);
}

std::vector<LayerRect> DRCChecker::getMinScope(const std::vector<DRCRect>& drc_rect_list)
{
  return getMinSpacingRect(convertToIDSRect(drc_rect_list));
}

#endif

#if 1  // 碰撞可能会产生DRC的最大膨胀矩形

std::vector<LayerRect> DRCChecker::getMaxScope(const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return getMaxScope(drc_rect_list);
}

std::vector<LayerRect> DRCChecker::getMaxScope(const std::vector<DRCRect>& drc_rect_list)
{
  return getMinSpacingRect(convertToIDSRect(drc_rect_list));
}

#endif

#if 1  // 获得违例信息

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getViolationInfo(RegionQuery& region_query,
                                                                               const std::vector<DRCRect>& drc_rect_list)
{
  RegionQuery* region_query_ref = &region_query;

  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;

  std::vector<ViolationInfo> violation_info_list;
  checkMinSpacingByOther(region_query_ref, drc_rect_list, violation_info_list);
  uniqueViolationInfoList(violation_info_list);
  for (ViolationInfo& violation_info : violation_info_list) {
    drc_violation_map[violation_info.get_rule_name()].push_back(violation_info);
  }
  return drc_violation_map;
}

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getViolationInfo(RegionQuery& region_query)
{
  RegionQuery* region_query_ref = &region_query;

  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;

  std::vector<ViolationInfo> violation_info_list;
  for (auto& [layer_idx, net_rect_list_map] : region_query.get_routing_net_rect_map()) {
    for (auto& [net_idx, rect_list] : net_rect_list_map) {
      for (const LayerRect& rect : rect_list) {
        checkMinSpacingByOther(region_query_ref, DRCRect(net_idx, rect, true), violation_info_list);
      }
    }
  }
  uniqueViolationInfoList(violation_info_list);
  for (ViolationInfo& violation_info : violation_info_list) {
    drc_violation_map[violation_info.get_rule_name()].push_back(violation_info);
  }
  return drc_violation_map;
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

void DRCChecker::addEnvRectList(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  auto& routing_net_rect_map = region_query->get_routing_net_rect_map();
  auto& cut_net_rect_map = region_query->get_cut_net_rect_map();
  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCRect& drc_rect : drc_rect_list) {
    irt_int net_idx = drc_rect.get_net_idx();
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
      routing_net_rect_map[layer_idx][net_idx].insert(layer_rect);
      routing_net_shape_map[layer_idx][net_idx][layer_rect].push_back(rq_shape);
      routing_region_map[layer_idx].insert(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape));
    } else {
      cut_net_rect_map[layer_idx][net_idx].insert(layer_rect);
      cut_net_shape_map[layer_idx][net_idx][layer_rect].push_back(rq_shape);
      cut_region_map[layer_idx].insert(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape));
    }
  }
}

void DRCChecker::delEnvRectList(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  auto& routing_net_rect_map = region_query->get_routing_net_rect_map();
  auto& cut_net_rect_map = region_query->get_cut_net_rect_map();
  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCRect& drc_rect : drc_rect_list) {
    irt_int net_idx = drc_rect.get_net_idx();
    const LayerRect layer_rect = drc_rect.get_layer_rect();
    irt_int layer_idx = layer_rect.get_layer_idx();

    RQShape* rq_shape = nullptr;
    if (drc_rect.get_is_routing()) {
      // 从routing_net_shape_map中删除数据
      if (!RTUtil::exist(routing_net_shape_map, layer_idx) || !RTUtil::exist(routing_net_shape_map[layer_idx], net_idx)
          || !RTUtil::exist(routing_net_shape_map[layer_idx][net_idx], layer_rect)
          || routing_net_shape_map[layer_idx][net_idx][layer_rect].empty()) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      std::vector<RQShape*>& rq_shape_list = routing_net_shape_map[layer_idx][net_idx][layer_rect];
      rq_shape = rq_shape_list.back();
      rq_shape_list.pop_back();
      if (rq_shape_list.empty()) {
        routing_net_shape_map[layer_idx][net_idx].erase(layer_rect);
        routing_net_rect_map[layer_idx][net_idx].erase(layer_rect);
      }
      // 从rtree中删除数据
      if (!RTUtil::exist(routing_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      if (!routing_region_map[layer_idx].remove(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape))) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
    } else {
      // 从cut_net_shape_map中删除数据
      if (!RTUtil::exist(cut_net_shape_map, layer_idx) || !RTUtil::exist(cut_net_shape_map[layer_idx], net_idx)
          || !RTUtil::exist(cut_net_shape_map[layer_idx][net_idx], layer_rect)
          || cut_net_shape_map[layer_idx][net_idx][layer_rect].empty()) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      std::vector<RQShape*>& rq_shape_list = cut_net_shape_map[layer_idx][net_idx][layer_rect];
      rq_shape = rq_shape_list.back();
      rq_shape_list.pop_back();
      if (rq_shape_list.empty()) {
        cut_net_shape_map[layer_idx][net_idx].erase(layer_rect);
        cut_net_rect_map[layer_idx][net_idx].erase(layer_rect);
      }
      // 从rtree中删除数据
      if (!RTUtil::exist(cut_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      if (!cut_region_map[layer_idx].remove(std::make_pair<>(rq_shape->get_enlarged_shape(), rq_shape))) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
    }
    // 释放资源
    delete rq_shape;
    rq_shape = nullptr;
  }
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

#if 1  // violation info

void DRCChecker::checkMinSpacingByOther(RegionQuery* region_query, const DRCRect& drc_rect, std::vector<ViolationInfo>& violation_info_list)
{
  std::vector<DRCRect> drc_rect_list = {drc_rect};
  checkMinSpacingByOther(region_query, drc_rect_list, violation_info_list);
}

void DRCChecker::checkMinSpacingByOther(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list,
                                        std::vector<ViolationInfo>& violation_info_list)
{
  for (const DRCRect& drc_rect : drc_rect_list) {
    RQShape drc_shape = convertToRQShape(drc_rect);
    irt_int layer_idx = drc_shape.get_routing_layer_idx();
    // 查询重叠
    std::vector<std::pair<BoostBox, RQShape*>> result_list;
    if (drc_shape.get_is_routing()) {
      auto& routing_region_map = region_query->get_routing_region_map();
      bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = routing_region_map[layer_idx];
      rtree.query(bgi::intersects(drc_shape.get_enlarged_shape()), std::back_inserter(result_list));
    } else {
      auto& cut_region_map = region_query->get_cut_region_map();
      bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = cut_region_map[layer_idx];
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
      PlanarRect enlarge_rect1 = RTUtil::getEnlargedRect(check_rect1, require_spacing);
      PlanarRect enlarge_rect2 = RTUtil::getEnlargedRect(check_rect2, require_spacing);
      if (!RTUtil::isOverlap(enlarge_rect1, check_rect2) && !RTUtil::isOverlap(enlarge_rect2, check_rect1)) {
        LOG_INST.error(Loc::current(), "Spacing violation rect is not overlap!");
      }

      LayerRect violation_region(RTUtil::getOverlap(enlarge_rect1, enlarge_rect2), layer_idx);

      std::map<irt_int, std::vector<LayerRect>> violation_net_shape_map;
      violation_net_shape_map[drc_shape.get_net_id()].emplace_back(check_rect1, layer_idx);
      violation_net_shape_map[overlap_shape->get_net_id()].emplace_back(check_rect2, layer_idx);

      ViolationInfo violation;
      violation.set_is_routing(drc_rect.get_is_routing());
      violation.set_rule_name("RT Spacing");
      violation.set_violation_region(violation_region);
      violation.set_net_shape_map(violation_net_shape_map);
      violation_info_list.push_back(violation);
    }
  }
}

void DRCChecker::uniqueViolationInfoList(std::vector<ViolationInfo>& violation_info_list)
{
  std::sort(violation_info_list.begin(), violation_info_list.end(), [](ViolationInfo& a, ViolationInfo& b) {
    if (a.get_is_routing() != b.get_is_routing()) {
      return a.get_is_routing();
    } else if (a.get_rule_name() != b.get_rule_name()) {
      return a.get_rule_name().size() < b.get_rule_name().size();
    } else if (a.get_violation_region() != b.get_violation_region()) {
      LayerRect& a_region = a.get_violation_region();
      LayerRect& b_region = b.get_violation_region();
      if (a_region.get_layer_idx() != b_region.get_layer_idx()) {
        return a_region.get_layer_idx() < b_region.get_layer_idx();
      } else if (a_region.get_lb() != b_region.get_lb()) {
        return CmpPlanarCoordByXASC()(a_region.get_lb(), b_region.get_lb());
      } else {
        return CmpPlanarCoordByXASC()(a_region.get_rt(), b_region.get_rt());
      }
    } else {
      std::set<irt_int> a_net_set;
      for (auto& [net_idx, rect_list] : a.get_net_shape_map()) {
        a_net_set.insert(net_idx);
      }
      std::set<irt_int> b_net_set;
      for (auto& [net_idx, rect_list] : b.get_net_shape_map()) {
        b_net_set.insert(net_idx);
      }
      return a_net_set < b_net_set;
    }
  });
  RTUtil::merge(violation_info_list, [](ViolationInfo& a, ViolationInfo& b) {
    std::set<irt_int> a_net_set;
    for (auto& [net_idx, rect_list] : a.get_net_shape_map()) {
      a_net_set.insert(net_idx);
    }
    std::set<irt_int> b_net_set;
    for (auto& [net_idx, rect_list] : b.get_net_shape_map()) {
      b_net_set.insert(net_idx);
    }
    return a_net_set == b_net_set && a.get_is_routing() == b.get_is_routing() && a.get_rule_name() == b.get_rule_name()
           && RTUtil::isClosedOverlap(a.get_violation_region(), b.get_violation_region());
  });
}

#endif

}  // namespace irt
