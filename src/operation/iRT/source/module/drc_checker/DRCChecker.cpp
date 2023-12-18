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

#include "DRCShape.hpp"
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

#if 1  // 获得DRCShapeList

std::vector<DRCShape> DRCChecker::getDRCShapeList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<DRCShape> drc_shape_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    for (DRCShape& drc_shape : getDRCShapeList(net_idx, segment)) {
      drc_shape_list.push_back(drc_shape);
    }
  }
  return drc_shape_list;
}

std::vector<DRCShape> DRCChecker::getDRCShapeList(irt_int net_idx, Segment<LayerCoord>& segment)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCShape> drc_shape_list;
  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  irt_int first_layer_idx = first_coord.get_layer_idx();
  irt_int second_layer_idx = second_coord.get_layer_idx();
  if (first_layer_idx != second_layer_idx) {
    RTUtil::swapByASC(first_layer_idx, second_layer_idx);
    for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
      ViaMaster& via_master = layer_via_master_list[layer_idx].front();

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
      drc_shape_list.emplace_back(net_idx, offset_above_enclosure, true);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
      drc_shape_list.emplace_back(net_idx, offset_below_enclosure, true);

      for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
        LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
        drc_shape_list.emplace_back(net_idx, offset_cut_shape, false);
      }
    }
  } else {
    irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
    LayerRect wire_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
    drc_shape_list.emplace_back(net_idx, wire_rect, true);
  }
  return drc_shape_list;
}

std::vector<DRCShape> DRCChecker::getDRCShapeList(irt_int net_idx, MTree<LayerCoord>& coord_tree)
{
  std::vector<DRCShape> drc_shape_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(coord_tree)) {
    LayerCoord first_coord = coord_segment.get_first()->value();
    LayerCoord second_coord = coord_segment.get_second()->value();
    Segment<LayerCoord> segment(first_coord, second_coord);
    for (DRCShape& drc_shape : getDRCShapeList(net_idx, segment)) {
      drc_shape_list.push_back(drc_shape);
    }
  }
  return drc_shape_list;
}

std::vector<DRCShape> DRCChecker::getDRCShapeList(irt_int net_idx, MTree<PhysicalNode>& physical_node_tree)
{
  std::vector<DRCShape> drc_shape_list;
  for (TNode<PhysicalNode>* physical_node_node : RTUtil::getNodeList(physical_node_tree)) {
    for (DRCShape& drc_shape : getDRCShapeList(net_idx, physical_node_node->value())) {
      drc_shape_list.push_back(drc_shape);
    }
  }
  return drc_shape_list;
}

std::vector<DRCShape> DRCChecker::getDRCShapeList(irt_int net_idx, PhysicalNode& physical_node)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCShape> drc_shape_list;
  if (physical_node.isType<WireNode>()) {
    WireNode& wire_node = physical_node.getNode<WireNode>();
    PlanarRect wire_rect = RTUtil::getEnlargedRect(wire_node.get_first(), wire_node.get_second(), wire_node.get_wire_width() / 2);
    drc_shape_list.emplace_back(net_idx, LayerRect(wire_rect, wire_node.get_layer_idx()), true);
  } else if (physical_node.isType<ViaNode>()) {
    ViaNode& via_node = physical_node.getNode<ViaNode>();
    ViaMasterIdx& via_master_idx = via_node.get_via_master_idx();
    ViaMaster& via_master = layer_via_master_list[via_master_idx.get_below_layer_idx()][via_master_idx.get_via_idx()];

    LayerRect& above_enclosure = via_master.get_above_enclosure();
    LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, via_node), above_enclosure.get_layer_idx());
    drc_shape_list.emplace_back(net_idx, offset_above_enclosure, true);

    LayerRect& below_enclosure = via_master.get_below_enclosure();
    LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, via_node), below_enclosure.get_layer_idx());
    drc_shape_list.emplace_back(net_idx, offset_below_enclosure, true);

    for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
      LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, via_node), via_master.get_cut_layer_idx());
      drc_shape_list.emplace_back(net_idx, offset_cut_shape, false);
    }
  } else if (physical_node.isType<PatchNode>()) {
    PatchNode& patch_node = physical_node.getNode<PatchNode>();
    drc_shape_list.emplace_back(net_idx, patch_node, true);
  }
  return drc_shape_list;
}

#endif

#if 1  // 返回RegionQuery中的形状map

std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>>& DRCChecker::getLayerInfoRectMap(
    RegionQuery& region_query, bool is_routing)
{
  if (is_routing) {
    return region_query.get_routing_info_rect_map();
  } else {
    return region_query.get_cut_info_rect_map();
  }
}

#endif

#if 1  // 更新RegionQuery的RectList

void DRCChecker::updateRectList(RegionQuery& region_query, ChangeType change_type, const DRCShape& drc_shape)
{
  std::vector<DRCShape> drc_shape_list{drc_shape};
  updateRectList(region_query, change_type, drc_shape_list);
}

void DRCChecker::updateRectList(RegionQuery& region_query, ChangeType change_type, const std::vector<DRCShape>& drc_shape_list)
{
  RegionQuery* region_query_ref = &region_query;

  if (change_type == ChangeType::kAdd) {
    addEnvRectList(region_query_ref, drc_shape_list);
  } else if (change_type == ChangeType::kDel) {
    delEnvRectList(region_query_ref, drc_shape_list);
  }
}

#endif

#if 1  // 碰撞一定会产生DRC的最小膨胀矩形

std::vector<LayerRect> DRCChecker::getMinScope(const DRCShape& drc_shape)
{
  std::vector<DRCShape> drc_shape_list{drc_shape};
  return getMinScope(drc_shape_list);
}

std::vector<LayerRect> DRCChecker::getMinScope(const std::vector<DRCShape>& drc_shape_list)
{
  return getMinSpacingRect(drc_shape_list);
}

#endif

#if 1  // 碰撞可能会产生DRC的最大膨胀矩形

std::vector<LayerRect> DRCChecker::getMaxScope(const DRCShape& drc_shape)
{
  std::vector<DRCShape> drc_shape_list{drc_shape};
  return getMaxScope(drc_shape_list);
}

std::vector<LayerRect> DRCChecker::getMaxScope(const std::vector<DRCShape>& drc_shape_list)
{
  return getMaxSpacingRect(drc_shape_list);
}

#endif

#if 1  // 获得违例信息

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getEnvViolationInfo(RegionQuery& region_query,
                                                                                  const std::vector<DRCCheckType>& check_type_list,
                                                                                  const std::vector<DRCShape>& drc_shape_list)
{
  irt_int enable_idrc_interface = DM_INST.getConfig().enable_idrc_interface;

  if (enable_idrc_interface == 0) {
    return getEnvViolationInfoByRT(region_query, check_type_list, drc_shape_list);
  } else {
    return getEnvViolationInfoByiDRC(region_query, check_type_list, drc_shape_list);
  }
}

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getSelfViolationInfo(const std::vector<DRCCheckType>& check_type_list,
                                                                                   const std::vector<DRCShape>& drc_shape_list)
{
  irt_int enable_idrc_interface = DM_INST.getConfig().enable_idrc_interface;

  if (enable_idrc_interface == 0) {
    return getSelfViolationInfoByRT(check_type_list, drc_shape_list);
  } else {
    return getSelfViolationInfoByiDRC(check_type_list, drc_shape_list);
  }
}

#endif

// private

DRCChecker* DRCChecker::_dc_instance = nullptr;

// function

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getEnvViolationInfoByRT(RegionQuery& region_query,
                                                                                      const std::vector<DRCCheckType>& check_type_list,
                                                                                      const std::vector<DRCShape>& drc_shape_list)
{
  RegionQuery* region_query_ref = &region_query;

  std::vector<ViolationInfo> violation_info_list;
  for (DRCCheckType check_type : check_type_list) {
    switch (check_type) {
      case DRCCheckType::kSpacing:
        checkMinSpacingByOther(region_query_ref, drc_shape_list, violation_info_list);
        break;
      default:
        LOG_INST.warn(Loc::current(), "Unsupported check type!");
        break;
    }
  }
  uniqueViolationInfoList(violation_info_list);

  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  for (ViolationInfo& violation_info : violation_info_list) {
    drc_violation_map[violation_info.get_rule_name()].push_back(violation_info);
  }
  return drc_violation_map;
}

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getEnvViolationInfoByiDRC(RegionQuery& region_query,
                                                                                        const std::vector<DRCCheckType>& check_type_list,
                                                                                        const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<BaseShape> base_shape_list;
  for (const DRCShape& drc_shape : drc_shape_list) {
    base_shape_list.push_back(convert(drc_shape));
  }

  std::map<std::string, std::vector<BaseViolationInfo>> base_violation_info_map
      = RTAPI_INST.getEnvViolationInfo(region_query.get_base_region(), check_type_list, base_shape_list);

  std::map<std::string, std::vector<ViolationInfo>> violation_info_map;
  for (auto& [rule_name, base_violation_info_list] : base_violation_info_map) {
    for (BaseViolationInfo& base_violation_info : base_violation_info_list) {
      violation_info_map[rule_name].push_back(convert(base_violation_info));
    }
  }

  return violation_info_map;
}

BaseShape DRCChecker::convert(const DRCShape& drc_shape)
{
  BaseShape base_shape;

  base_shape.set_base_info(drc_shape.get_base_info());
  base_shape.set_shape(RTUtil::convertToBGRectInt(drc_shape.get_layer_rect().get_rect()));
  base_shape.set_is_routing(drc_shape.get_is_routing());
  base_shape.set_layer_idx(drc_shape.get_layer_rect().get_layer_idx());

  return base_shape;
}

ViolationInfo DRCChecker::convert(BaseViolationInfo& base_violation_info)
{
  ViolationInfo violation_info;

  violation_info.set_rule_name(base_violation_info.get_rule_name());
  violation_info.set_violation_region(
      LayerRect(RTUtil::convertToPlanarRect(base_violation_info.get_violation_region()), base_violation_info.get_layer_idx()));
  violation_info.set_is_routing(base_violation_info.get_is_routing());
  violation_info.set_base_info_set(base_violation_info.get_base_info_set());

  return violation_info;
}

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getSelfViolationInfoByRT(const std::vector<DRCCheckType>& check_type_list,
                                                                                       const std::vector<DRCShape>& drc_shape_list)
{
  // RegionQuery* region_query_ref = &region_query;

  std::vector<ViolationInfo> violation_info_list;
  for (DRCCheckType check_type : check_type_list) {
    switch (check_type) {
      case DRCCheckType::kMinArea:
        checkMinArea(drc_shape_list, violation_info_list);
        break;
      case DRCCheckType::kMinStep:
        // checkMinStep(drc_shape_list, violation_info_list);
        break;
      default:
        LOG_INST.warn(Loc::current(), "Unsupported check type!");
        break;
    }
  }
  uniqueViolationInfoList(violation_info_list);

  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  for (ViolationInfo& violation_info : violation_info_list) {
    drc_violation_map[violation_info.get_rule_name()].push_back(violation_info);
  }
  return drc_violation_map;
}

std::map<std::string, std::vector<ViolationInfo>> DRCChecker::getSelfViolationInfoByiDRC(const std::vector<DRCCheckType>& check_type_list,
                                                                                         const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<BaseShape> base_shape_list;
  for (const DRCShape& drc_shape : drc_shape_list) {
    base_shape_list.push_back(convert(drc_shape));
  }

  std::map<std::string, std::vector<BaseViolationInfo>> base_violation_info_map
      = RTAPI_INST.getSelfViolationInfo(check_type_list, base_shape_list);

  std::map<std::string, std::vector<ViolationInfo>> violation_info_map;
  for (auto& [rule_name, base_violation_info_list] : base_violation_info_map) {
    for (BaseViolationInfo& base_violation_info : base_violation_info_list) {
      violation_info_map[rule_name].push_back(convert(base_violation_info));
    }
  }

  return violation_info_map;
}

void DRCChecker::addEnvRectList(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  auto& routing_info_rect_map = region_query->get_routing_info_rect_map();
  auto& cut_info_rect_map = region_query->get_cut_info_rect_map();
  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCShape& drc_shape : drc_shape_list) {
    const LayerRect& layer_rect = drc_shape.get_layer_rect();
    BGRectInt shape = RTUtil::convertToBGRectInt(layer_rect);
    const BaseInfo& info = drc_shape.get_base_info();
    irt_int layer_idx = layer_rect.get_layer_idx();
    irt_int net_idx = info.get_net_idx();

    BaseShape* base_shape = new BaseShape();
    base_shape->set_base_info(drc_shape.get_base_info());
    base_shape->set_shape(shape);
    base_shape->set_layer_idx(layer_idx);

    if (drc_shape.get_is_routing()) {
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(layer_rect);
      BGRectInt enlarged_shape = RTUtil::enlargeBGRectInt(shape, min_spacing);
      routing_info_rect_map[layer_idx][info].insert(layer_rect);
      routing_net_shape_map[layer_idx][net_idx][layer_rect].push_back(base_shape);
      routing_region_map[layer_idx].insert(std::make_pair<>(enlarged_shape, base_shape));
    } else {
      irt_int min_spacing = cut_layer_list[layer_idx].getMinSpacing(layer_rect);
      BGRectInt enlarged_shape = RTUtil::enlargeBGRectInt(shape, min_spacing);
      cut_info_rect_map[layer_idx][info].insert(layer_rect);
      cut_net_shape_map[layer_idx][net_idx][layer_rect].push_back(base_shape);
      cut_region_map[layer_idx].insert(std::make_pair<>(enlarged_shape, base_shape));
    }
  }
}

void DRCChecker::delEnvRectList(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  auto& routing_info_rect_map = region_query->get_routing_info_rect_map();
  auto& cut_info_rect_map = region_query->get_cut_info_rect_map();
  auto& routing_net_shape_map = region_query->get_routing_net_shape_map();
  auto& cut_net_shape_map = region_query->get_cut_net_shape_map();
  auto& routing_region_map = region_query->get_routing_region_map();
  auto& cut_region_map = region_query->get_cut_region_map();

  for (const DRCShape& drc_shape : drc_shape_list) {
    const LayerRect layer_rect = drc_shape.get_layer_rect();
    const BaseInfo& info = drc_shape.get_base_info();
    irt_int layer_idx = layer_rect.get_layer_idx();
    irt_int net_idx = info.get_net_idx();

    BaseShape* base_shape = nullptr;
    if (drc_shape.get_is_routing()) {
      // 从routing_net_shape_map中删除数据
      if (!RTUtil::exist(routing_net_shape_map, layer_idx) || !RTUtil::exist(routing_net_shape_map[layer_idx], net_idx)
          || !RTUtil::exist(routing_net_shape_map[layer_idx][net_idx], layer_rect)
          || routing_net_shape_map[layer_idx][net_idx][layer_rect].empty()) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      std::vector<BaseShape*>& base_shape_list = routing_net_shape_map[layer_idx][net_idx][layer_rect];
      base_shape = base_shape_list.back();
      base_shape_list.pop_back();
      if (base_shape_list.empty()) {
        routing_net_shape_map[layer_idx][net_idx].erase(layer_rect);
        routing_info_rect_map[layer_idx][info].erase(layer_rect);
      }
      // 从rtree中删除数据
      if (!RTUtil::exist(routing_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(layer_rect);
      PlanarRect enlarged_rect = RTUtil::getEnlargedRect(layer_rect, min_spacing);
      if (!routing_region_map[layer_idx].remove(std::make_pair<>(RTUtil::convertToBGRectInt(enlarged_rect), base_shape))) {
        // LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
    } else {
      // 从cut_net_shape_map中删除数据
      if (!RTUtil::exist(cut_net_shape_map, layer_idx) || !RTUtil::exist(cut_net_shape_map[layer_idx], net_idx)
          || !RTUtil::exist(cut_net_shape_map[layer_idx][net_idx], layer_rect)
          || cut_net_shape_map[layer_idx][net_idx][layer_rect].empty()) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      std::vector<BaseShape*>& base_shape_list = cut_net_shape_map[layer_idx][net_idx][layer_rect];
      base_shape = base_shape_list.back();
      base_shape_list.pop_back();
      if (base_shape_list.empty()) {
        cut_net_shape_map[layer_idx][net_idx].erase(layer_rect);
        cut_info_rect_map[layer_idx][info].erase(layer_rect);
      }
      // 从rtree中删除数据
      if (!RTUtil::exist(cut_region_map, layer_idx)) {
        LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
      irt_int min_spacing = cut_layer_list[layer_idx].getMinSpacing(layer_rect);
      PlanarRect enlarged_rect = RTUtil::getEnlargedRect(layer_rect, min_spacing);
      if (!cut_region_map[layer_idx].remove(std::make_pair<>(RTUtil::convertToBGRectInt(enlarged_rect), base_shape))) {
        // LOG_INST.error(Loc::current(), "There is no rect in net_", net_idx, "!");
      }
    }
    // 释放资源
    delete base_shape;
    base_shape = nullptr;
  }
}

std::vector<LayerRect> DRCChecker::getMinSpacingRect(const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  std::vector<LayerRect> min_scope_list;
  for (const DRCShape& drc_shape : drc_shape_list) {
    irt_int layer_idx = drc_shape.get_layer_rect().get_layer_idx();
    PlanarRect rect = drc_shape.get_layer_rect();
    if (drc_shape.get_is_routing()) {
      min_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, routing_layer_list[layer_idx].getMinSpacing(rect)), layer_idx);
    } else {
      min_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, cut_layer_list[layer_idx].getMinSpacing(rect)), layer_idx);
    }
  }
  return min_scope_list;
}

std::vector<LayerRect> DRCChecker::getMaxSpacingRect(const std::vector<DRCShape>& drc_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  std::vector<LayerRect> max_scope_list;
  for (const DRCShape& drc_shape : drc_shape_list) {
    irt_int layer_idx = drc_shape.get_layer_rect().get_layer_idx();
    PlanarRect rect = drc_shape.get_layer_rect();
    if (drc_shape.get_is_routing()) {
      max_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, routing_layer_list[layer_idx].getMaxSpacing(rect)), layer_idx);
    } else {
      max_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, cut_layer_list[layer_idx].getMaxSpacing(rect)), layer_idx);
    }
  }
  return max_scope_list;
}

#if 1  // violation info

void DRCChecker::checkMinSpacingByOther(RegionQuery* region_query, const DRCShape& drc_shape,
                                        std::vector<ViolationInfo>& violation_info_list)
{
  std::vector<DRCShape> drc_shape_list = {drc_shape};
  checkMinSpacingByOther(region_query, drc_shape_list, violation_info_list);
}

void DRCChecker::checkMinSpacingByOther(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list,
                                        std::vector<ViolationInfo>& violation_info_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  for (const DRCShape& drc_shape : drc_shape_list) {
    const LayerRect& layer_rect = drc_shape.get_layer_rect();
    BaseShape base_shape = convert(drc_shape);
    irt_int layer_idx = base_shape.get_layer_idx();
    irt_int base_net_idx = base_shape.get_base_info().get_net_idx();

    // 查询重叠
    irt_int base_region_min_spacing = -1;
    std::vector<std::pair<BGRectInt, BaseShape*>> result_list;
    if (base_shape.get_is_routing()) {
      base_region_min_spacing = routing_layer_list[layer_idx].getMinSpacing(layer_rect);
      PlanarRect enlarged_rect = RTUtil::getEnlargedRect(layer_rect, base_region_min_spacing);
      auto& routing_region_map = region_query->get_routing_region_map();
      bgi::rtree<std::pair<BGRectInt, BaseShape*>, bgi::quadratic<16UL>>& rtree = routing_region_map[layer_idx];
      rtree.query(bgi::intersects(RTUtil::convertToBGRectInt(enlarged_rect)), std::back_inserter(result_list));
    } else {
      base_region_min_spacing = cut_layer_list[layer_idx].getMinSpacing(layer_rect);
      PlanarRect enlarged_rect = RTUtil::getEnlargedRect(layer_rect, base_region_min_spacing);
      auto& cut_region_map = region_query->get_cut_region_map();
      bgi::rtree<std::pair<BGRectInt, BaseShape*>, bgi::quadratic<16UL>>& rtree = cut_region_map[layer_idx];
      rtree.query(bgi::intersects(RTUtil::convertToBGRectInt(enlarged_rect)), std::back_inserter(result_list));
    }

    // 遍历每个重叠 判断是否Spacing违例
    for (size_t i = 0; i < result_list.size(); i++) {
      BaseShape* overlap_shape = result_list[i].second;
      irt_int overlap_net_idx = overlap_shape->get_base_info().get_net_idx();
      if (overlap_net_idx == base_net_idx) {
        continue;
      }
      irt_int overlap_shape_min_spacing = -1;
      PlanarRect overlap_rect = RTUtil::convertToPlanarRect(overlap_shape->get_shape());
      if (overlap_shape->get_is_routing()) {
        overlap_shape_min_spacing = routing_layer_list[layer_idx].getMinSpacing(overlap_rect);
      } else {
        overlap_shape_min_spacing = cut_layer_list[layer_idx].getMinSpacing(overlap_rect);
      }

      irt_int require_spacing = std::max(base_region_min_spacing, overlap_shape_min_spacing);
      irt_int spacing = RTUtil::getEuclideanDistance(overlap_shape->get_shape(), base_shape.get_shape());
      if (spacing >= require_spacing) {
        continue;
      }

      PlanarRect check_rect1 = RTUtil::convertToPlanarRect(base_shape.get_shape());
      PlanarRect check_rect2 = RTUtil::convertToPlanarRect(overlap_shape->get_shape());
      PlanarRect enlarge_rect1 = RTUtil::getEnlargedRect(check_rect1, require_spacing);
      PlanarRect enlarge_rect2 = RTUtil::getEnlargedRect(check_rect2, require_spacing);
      if (!RTUtil::isOverlap(enlarge_rect1, check_rect2) && !RTUtil::isOverlap(enlarge_rect2, check_rect1)) {
        // LOG_INST.error(Loc::current(), "Spacing violation rect is not overlap!");
      }
      std::string rule_name;
      if (drc_shape.get_is_routing()) {
        rule_name = RTUtil::isOverlap(check_rect1, check_rect2) ? "Metal Short" : "Metal Spacing";
      } else {
        rule_name = RTUtil::isOverlap(check_rect1, check_rect2) ? "Cut Short" : "Cut Spacing";
      }
      if (!RTUtil::isOverlap(enlarge_rect1, enlarge_rect2)) {
        continue;
      }
      LayerRect violation_region(RTUtil::getOverlap(enlarge_rect1, enlarge_rect2), layer_idx);

      std::set<BaseInfo, CmpBaseInfo> base_info_set;
      base_info_set.insert(base_shape.get_base_info());
      base_info_set.insert(overlap_shape->get_base_info());

      ViolationInfo violation;
      violation.set_is_routing(drc_shape.get_is_routing());
      violation.set_rule_name(rule_name);
      violation.set_violation_region(violation_region);
      violation.set_base_info_set(base_info_set);
      violation_info_list.push_back(violation);
    }
  }
}

void DRCChecker::checkMinArea(const std::vector<DRCShape>& drc_shape_list, std::vector<ViolationInfo>& violation_info_list)
{
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::map<irt_int, std::map<irt_int, GTLPolySetInt>> layer_net_polgon_set_map;
  for (const DRCShape& drc_shape : drc_shape_list) {
    if (!drc_shape.get_is_routing()) {
      continue;
    }
    irt_int net_idx = drc_shape.get_base_info().get_net_idx();
    irt_int layer_idx = drc_shape.get_layer_rect().get_layer_idx();
    if (bottom_routing_layer_idx <= layer_idx && layer_idx <= top_routing_layer_idx) {
      GTLPolySetInt& poly_set = layer_net_polgon_set_map[layer_idx][net_idx];
      poly_set += RTUtil::convertToGTLRectInt(drc_shape.get_layer_rect());
    }
  }

  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  for (auto& [layer_idx, net_poly_set_map] : layer_net_polgon_set_map) {
    for (auto& [net_idx, poly_set] : net_poly_set_map) {
      std::vector<GTLPolyInt> poly_list;
      poly_set.get_polygons(poly_list);
      for (GTLPolyInt& poly : poly_list) {
        if (gtl::area(poly) >= routing_layer_list[layer_idx].get_min_area()) {
          continue;
        }
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_rectangles(gtl_rect_list, poly);

        std::map<irt_int, std::vector<LayerRect>> violation_net_shape_map;
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          violation_net_shape_map[net_idx].emplace_back(RTUtil::convertToPlanarRect(gtl_rect), layer_idx);
        }
        LayerRect violation_region = violation_net_shape_map[net_idx].front();

        BaseInfo base_info;
        base_info.set_net_idx(net_idx);

        std::set<BaseInfo, CmpBaseInfo> base_info_set;
        base_info_set.insert(base_info);

        ViolationInfo violation;
        violation.set_is_routing(true);
        violation.set_rule_name("Min Area");
        violation.set_violation_region(violation_region);
        // violation.set_base_info_set(base_info_set);
        violation_info_list.push_back(violation);
      }
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
      for (const BaseInfo& base_info : a.get_base_info_set()) {
        a_net_set.insert(base_info.get_net_idx());
      }
      std::set<irt_int> b_net_set;
      for (const BaseInfo& base_info : b.get_base_info_set()) {
        b_net_set.insert(base_info.get_net_idx());
      }
      return a_net_set < b_net_set;
    }
  });
  RTUtil::merge(violation_info_list, [](ViolationInfo& a, ViolationInfo& b) {
    std::set<irt_int> a_net_set;
    for (const BaseInfo& base_info : a.get_base_info_set()) {
      a_net_set.insert(base_info.get_net_idx());
    }
    std::set<irt_int> b_net_set;
    for (const BaseInfo& base_info : b.get_base_info_set()) {
      b_net_set.insert(base_info.get_net_idx());
    }
    return a_net_set == b_net_set && a.get_is_routing() == b.get_is_routing() && a.get_rule_name() == b.get_rule_name()
           && RTUtil::isClosedOverlap(a.get_violation_region(), b.get_violation_region());
  });
}

#endif

}  // namespace irt
