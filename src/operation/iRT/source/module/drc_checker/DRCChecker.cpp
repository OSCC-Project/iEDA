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
#include "RTAPI.hpp"
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
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
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
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, MTree<PHYNode>& phy_node_tree)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(phy_node_tree)) {
    PHYNode& phy_node = phy_node_node->value();
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
  }
  return drc_rect_list;
}

// void* DRCChecker::initRegionQuery()
// {
//   void* region_query = nullptr;
//   if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
//     region_query = RTAPI_INST.initRegionQuery();
//   } else {
//     region_query = new RegionQuery();
//   }
//   return region_query;
// }

std::map<irt_int, std::map<irt_int, std::set<LayerRect, CmpLayerRectByXASC>>> DRCChecker::getRoutingNetRectMap(void* region_query,
                                                                                                               bool is_routing)
{
  std::map<irt_int, std::map<irt_int, std::set<LayerRect, CmpLayerRectByXASC>>> routing_net_rect_map;
  return routing_net_rect_map;
}

void DRCChecker::addEnvRectList(void* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  addEnvRectList(region_query, drc_rect_list);
}

void DRCChecker::addEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    RTAPI_INST.addEnvRectList(region_query, ids_rect_list);
  } else {
    addEnvRectListByRTDRC(region_query, ids_rect_list);
  }
}

void DRCChecker::delEnvRectList(void* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  delEnvRectList(region_query, drc_rect_list);
}

void DRCChecker::delEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    RTAPI_INST.delEnvRectList(region_query, ids_rect_list);
  } else {
    delEnvRectListByRTDRC(region_query, ids_rect_list);
  }
}

bool DRCChecker::hasViolation(void* region_query, const DRCRect& drc_rect)
{
  std::vector<DRCRect> drc_rect_list{drc_rect};
  return hasViolation(region_query, drc_rect_list);
}

bool DRCChecker::hasViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  bool has_violation = false;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    has_violation = RTAPI_INST.hasViolation(region_query, ids_rect_list);
  } else {
    for (auto [drc, num] : getViolationByRTDRC(region_query, ids_rect_list)) {
      if (num > 0) {
        return true;
      }
    }
    return false;
  }
  return has_violation;
}

std::map<std::string, int> DRCChecker::getViolation(void* region_query)
{
  std::map<std::string, irt_int> violation_name_num_map;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    violation_name_num_map = RTAPI_INST.getViolation(region_query);
  } else {
    violation_name_num_map = getViolationByRTDRC(region_query);
  }
  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);

  std::map<std::string, irt_int> violation_name_num_map;
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    violation_name_num_map = RTAPI_INST.getViolation(region_query, ids_rect_list);
  } else {
    violation_name_num_map = getViolationByRTDRC(region_query, ids_rect_list);
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
    min_scope_list = RTAPI_INST.getMaxScope(ids_rect_list);
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

void DRCChecker::plotRegionQuery(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
  if (DM_INST.getConfig().enable_idrc_interfaces == 1) {
    return;
  } else {
    RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
    std::vector<ids::DRCRect> ids_rect_list = convertToIDSRect(drc_rect_list);
    plotRegionQueryByRTDRC(rt_region_query, ids_rect_list);
  }
}

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

void DRCChecker::addEnvRectListByRTDRC(void* region_query, const std::vector<ids::DRCRect>& env_rect_list)
{
  RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
  auto& obj_id_shape_map = rt_region_query->get_obj_id_shape_map();
  auto& region_map = rt_region_query->get_region_map();

  std::vector<RQShape> rq_shape_list = getRQShapeList(env_rect_list);
  for (RQShape& rq_shape : rq_shape_list) {
    obj_id_shape_map[rq_shape.get_net_id()].push_back(rq_shape);
  }
  for (auto& [net_id, rq_shape_list] : obj_id_shape_map) {
    for (RQShape& rq_shape : rq_shape_list) {
      region_map[rq_shape.get_routing_layer_idx()].insert(std::make_pair<>(rq_shape.get_enlarged_shape(), &rq_shape));
    }
  }
}

std::vector<RQShape> DRCChecker::getRQShapeList(const std::vector<ids::DRCRect>& env_rect_list)
{
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = DM_INST.getHelper().get_routing_layer_name_to_idx_map();

  std::vector<RQShape> rq_shape_list;
  for (ids::DRCRect env_rect : env_rect_list) {
    if (!RTUtil::exist(routing_layer_name_to_idx_map, env_rect.layer_name)) {
      continue;
    }
    RQShape rq_shape;

    BoostBox shape = convertBoostBox(env_rect);
    irt_int layer_idx = DM_INST.getHelper().getRoutingLayerIdxByName(env_rect.layer_name);
    RoutingLayer& routing_layer = DM_INST.getDatabase().get_routing_layer_list()[layer_idx];
    irt_int min_spacing = routing_layer.getMinSpacing(PlanarRect(env_rect.lb_x, env_rect.lb_y, env_rect.rt_x, env_rect.rt_y));
    BoostBox enlarged_shape = RTUtil::enlargeBoostBox(shape, min_spacing);

    rq_shape.set_shape(shape);
    rq_shape.set_net_id(env_rect.so_id);
    rq_shape.set_routing_layer_idx(layer_idx);
    rq_shape.set_min_spacing(min_spacing);
    rq_shape.set_enlarged_shape(enlarged_shape);
    rq_shape_list.push_back(rq_shape);
  }
  return rq_shape_list;
}

BoostBox DRCChecker::convertBoostBox(ids::DRCRect ids_rect)
{
  return BoostBox(BoostPoint(ids_rect.lb_x, ids_rect.lb_y), BoostPoint(ids_rect.rt_x, ids_rect.rt_y));
}

void DRCChecker::delEnvRectListByRTDRC(void* region_query, const std::vector<ids::DRCRect>& env_rect_list)
{
  RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
  auto& obj_id_shape_map = rt_region_query->get_obj_id_shape_map();
  auto& region_map = rt_region_query->get_region_map();

  std::set<irt_int> del_net_id_set;
  for (const ids::DRCRect& ids_rect : env_rect_list) {
    del_net_id_set.insert(ids_rect.so_id);
  }
  for (irt_int net_id : del_net_id_set) {
    if (!RTUtil::exist(obj_id_shape_map, net_id)) {
      LOG_INST.warning(Loc::current(), "Net id : ", net_id, " no exist!");
      continue;
    }
    // 从rtree中删除数据
    for (RQShape& rq_shape : obj_id_shape_map[net_id]) {
      region_map[rq_shape.get_routing_layer_idx()].remove(std::make_pair<>(rq_shape.get_enlarged_shape(), &rq_shape));
    }
    // 从obj map中删除数据
    obj_id_shape_map.erase(net_id);
  }
}

std::map<std::string, int> DRCChecker::getViolationByRTDRC(void* region_query)
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

  RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
  auto& obj_id_shape_map = rt_region_query->get_obj_id_shape_map();

  for (auto& [net_id, shape_list] : obj_id_shape_map) {
    // check drc by other
    for (auto [violation_name, num] : checkByOtherByRTDRC(region_query, shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
    // check drc by self
    for (auto [violation_name, num] : checkBySelfByRTDRC(region_query, shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
  }

  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::getViolationByRTDRC(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list)
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
  for (RQShape& rq_shape : getRQShapeList(drc_rect_list)) {
    net_shape_list_map[rq_shape.get_net_id()].push_back(rq_shape);
  }
  for (auto& [net_id, shape_list] : net_shape_list_map) {
    // check drc by other
    for (auto [violation_name, num] : checkByOtherByRTDRC(region_query, shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
    // check drc by self
    // for (auto [violation_name, num] : checkBySelfByRTDRC(shape_list)) {
    //   violation_name_num_map[violation_name] += num;
    // }
  }

  return violation_name_num_map;
}

std::map<std::string, int> DRCChecker::checkByOtherByRTDRC(void* region_query, std::vector<RQShape>& drc_shape_list)
{
  RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
  auto& region_map = rt_region_query->get_region_map();

  std::map<std::string, int> violation_name_num_map;
  for (RQShape& drc_shape : drc_shape_list) {
    bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = region_map[drc_shape.get_routing_layer_idx()];

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

std::map<std::string, int> DRCChecker::checkBySelfByRTDRC(void* region_query, std::vector<RQShape>& drc_shape_list)
{
  RegionQuery* rt_region_query = static_cast<RegionQuery*>(region_query);
  auto& region_map = rt_region_query->get_region_map();

  std::map<std::string, int> violation_name_num_map;

  std::map<irt_int, std::vector<RQShape>> layer_net_shape_list_map;
  for (RQShape& drc_shape : drc_shape_list) {
    irt_int layer_idx = drc_shape.get_routing_layer_idx();
    layer_net_shape_list_map[layer_idx].push_back(drc_shape);

    bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = region_map[layer_idx];

    // 查询重叠
    std::vector<std::pair<BoostBox, RQShape*>> result_list;
    rtree.query(bgi::intersects(drc_shape.get_enlarged_shape()), std::back_inserter(result_list));

    for (size_t i = 0; i < result_list.size(); i++) {
      RQShape* overlap_shape = result_list[i].second;
      if (overlap_shape->get_net_id() != drc_shape.get_net_id()) {
        continue;
      }
      layer_net_shape_list_map[layer_idx].push_back(*overlap_shape);
    }
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
        violation_name_num_map["RT: Self drc"]++;
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

  auto& obj_id_shape_map = region_query->get_obj_id_shape_map();

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
  GPStruct box_scale_axis_struct("scale_axis");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
      for (irt_int x = x_grid.get_start_line(); x <= x_grid.get_end_line(); x += x_grid.get_step_length()) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kScaleAxis));
        gp_path.set_segment(x, die.get_real_lb_y(), x, die.get_real_rt_y());
        gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        box_scale_axis_struct.push(gp_path);
      }
    }
    for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
      for (irt_int y = y_grid.get_start_line(); y <= y_grid.get_end_line(); y += y_grid.get_step_length()) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kScaleAxis));
        gp_path.set_segment(die.get_real_lb_x(), y, die.get_real_rt_x(), y);
        gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        box_scale_axis_struct.push(gp_path);
      }
    }
  }
  gp_gds.addStruct(box_scale_axis_struct);

  // env shape
  for (auto& [net_id, shape_list] : obj_id_shape_map) {
    GPStruct net_shape_struct(RTUtil::getString("env shape(net_", net_id, ")"));
    GPGraphType type = net_id == -1 ? GPGraphType::kBlockage : GPGraphType::kPanelResult;
    for (RQShape& shape : shape_list) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<irt_int>(type));
      gp_boundary.set_rect(RTUtil::convertToPlanarRect(shape.get_shape()));
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(shape.get_routing_layer_idx()));
      net_shape_struct.push(gp_boundary);
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

}  // namespace irt
