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

#include "RegionQuery.hpp"

#include "DataManager.hpp"
#include "GDSPlotter.hpp"

namespace irt {

// private
void RegionQuery::init()
{
  _region_map.resize(DM_INST.getDatabase().get_routing_layer_list().size());
}

void RegionQuery::addEnvRectList(const ids::DRCRect& env_rect)
{
  std::vector<ids::DRCRect> env_rect_list{env_rect};
  addEnvRectList(env_rect_list);
}

void RegionQuery::addEnvRectList(const std::vector<ids::DRCRect>& env_rect_list)
{
  std::vector<RQShape> rq_shape_list = getRQShapeList(env_rect_list);
  for (RQShape& rq_shape : rq_shape_list) {
    _obj_id_shape_map[rq_shape.get_net_id()].push_back(rq_shape);
  }
  for (auto& [net_id, rq_shape_list] : _obj_id_shape_map) {
    for (RQShape& rq_shape : rq_shape_list) {
      _region_map[rq_shape.get_routing_layer_idx()].insert(std::make_pair<>(rq_shape.get_enlarged_shape(), &rq_shape));
    }
  }
}

std::vector<RQShape> RegionQuery::getRQShapeList(const std::vector<ids::DRCRect>& env_rect_list)
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

BoostBox RegionQuery::convertBoostBox(ids::DRCRect ids_rect)
{
  return BoostBox(BoostPoint(ids_rect.lb_x, ids_rect.lb_y), BoostPoint(ids_rect.rt_x, ids_rect.rt_y));
}

void RegionQuery::delEnvRectList(const ids::DRCRect& env_rect)
{
  std::vector<ids::DRCRect> env_rect_list{env_rect};
  delEnvRectList(env_rect_list);
}

void RegionQuery::delEnvRectList(const std::vector<ids::DRCRect>& env_rect_list)
{
  std::set<irt_int> del_net_id_set;
  for (const ids::DRCRect& ids_rect : env_rect_list) {
    del_net_id_set.insert(ids_rect.so_id);
  }
  for (irt_int net_id : del_net_id_set) {
    if (!RTUtil::exist(_obj_id_shape_map, net_id)) {
      LOG_INST.warning(Loc::current(), "Net id : ", net_id, " no exist!");
      continue;
    }
    // 从rtree中删除数据
    for (RQShape& rq_shape : _obj_id_shape_map[net_id]) {
      _region_map[net_id].remove(std::make_pair<>(rq_shape.get_enlarged_shape(), &rq_shape));
    }
    // 从obj map中删除数据
    _obj_id_shape_map.erase(net_id);
  }
}

bool RegionQuery::hasViolation(const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return hasViolation(drc_rect_list);
}

bool RegionQuery::hasViolation(const std::vector<ids::DRCRect>& drc_rect_list)
{
  for (auto [drc, num] : getViolation(drc_rect_list)) {
    if (num > 0) {
      return true;
    }
  }
  return false;
}

std::map<std::string, int> RegionQuery::getViolation(const std::vector<ids::DRCRect>& drc_rect_list)
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
    for (auto [violation_name, num] : checkByOther(shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
    // check drc by self
    // for (auto [violation_name, num] : checkBySelf(shape_list)) {
    //   violation_name_num_map[violation_name] += num;
    // }
  }

  return violation_name_num_map;
}

std::map<std::string, int> RegionQuery::checkByOther(std::vector<RQShape>& drc_shape_list)
{
  std::map<std::string, int> violation_name_num_map;
  for (RQShape& drc_shape : drc_shape_list) {
    bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = _region_map[drc_shape.get_routing_layer_idx()];

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

std::map<std::string, int> RegionQuery::checkBySelf(std::vector<RQShape>& drc_shape_list)
{
  std::map<std::string, int> violation_name_num_map;

  std::map<irt_int, std::vector<RQShape>> layer_net_shape_list_map;
  for (RQShape& drc_shape : drc_shape_list) {
    irt_int layer_idx = drc_shape.get_routing_layer_idx();
    layer_net_shape_list_map[layer_idx].push_back(drc_shape);

    bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16UL>>& rtree = _region_map[layer_idx];

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
        if (checkMinSpacing(net_shape1, net_shape2, net_shape_list)) {
          continue;
        }
        violation_name_num_map["RT: Self drc"]++;
      }
    }
  }
  return violation_name_num_map;
}

bool RegionQuery::checkMinSpacing(RQShape& net_shape1, RQShape& net_shape2, std::vector<RQShape>& net_shape_list)
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

std::map<std::string, int> RegionQuery::getViolation()
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

  for (auto& [net_id, shape_list] : _obj_id_shape_map) {
    // check drc by other
    for (auto [violation_name, num] : checkByOther(shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
    // check drc by self
    for (auto [violation_name, num] : checkBySelf(shape_list)) {
      violation_name_num_map[violation_name] += num;
    }
  }

  return violation_name_num_map;
}

std::vector<LayerRect> RegionQuery::getMaxScope(const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return getMaxScope(drc_rect_list);
}

std::vector<LayerRect> RegionQuery::getMaxScope(const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = DM_INST.getHelper().get_routing_layer_name_to_idx_map();

  std::vector<LayerRect> max_scope_list;
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    if (!RTUtil::exist(routing_layer_name_to_idx_map, drc_rect.layer_name)) {
      continue;
    }
    PlanarRect rect(drc_rect.lb_x, drc_rect.lb_y, drc_rect.rt_x, drc_rect.rt_y);
    irt_int layer_idx = DM_INST.getHelper().getRoutingLayerIdxByName(drc_rect.layer_name);
    RoutingLayer& routing_layer = DM_INST.getDatabase().get_routing_layer_list()[layer_idx];
    max_scope_list.emplace_back(RTUtil::getEnlargedRect(rect, routing_layer.getMinSpacing(rect)), layer_idx);
  }
  return max_scope_list;
}

std::vector<LayerRect> RegionQuery::getMinScope(const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return getMinScope(drc_rect_list);
}

std::vector<LayerRect> RegionQuery::getMinScope(const std::vector<ids::DRCRect>& drc_rect_list)
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

LayerRect RegionQuery::convertToLayerRect(ids::DRCRect ids_rect)
{
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = DM_INST.getHelper().get_routing_layer_name_to_idx_map();
  std::map<std::string, irt_int>& cut_layer_name_to_idx_map = DM_INST.getHelper().get_cut_layer_name_to_idx_map();

  LayerRect rt_rect;
  rt_rect.set_rect(ids_rect.lb_x, ids_rect.lb_y, ids_rect.rt_x, ids_rect.rt_y);

  if (RTUtil::exist(routing_layer_name_to_idx_map, ids_rect.layer_name)) {
    rt_rect.set_layer_idx(routing_layer_name_to_idx_map[ids_rect.layer_name]);
  } else if (RTUtil::exist(cut_layer_name_to_idx_map, ids_rect.layer_name)) {
    rt_rect.set_layer_idx(routing_layer_name_to_idx_map[ids_rect.layer_name]);
  }

  return rt_rect;
}

// plot region
void RegionQuery::plotRegionQuery(const std::vector<ids::DRCRect>& drc_rect_list)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string gp_temp_directory_path = DM_INST.getConfig().gp_temp_directory_path;

  irt_int width = INT_MAX;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
      width = std::min(width, x_grid.get_step_length());
    }
    for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
      width = std::min(width, y_grid.get_step_length());
    }
  }
  width = std::max(1, width / 3);

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
  for (auto& [net_id, shape_list] : _obj_id_shape_map) {
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