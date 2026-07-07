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
#include "RuleValidator.hpp"

#include "DRCHeader.hpp"
#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "PlanarRect.hpp"
#include "RVCluster.hpp"
#include "Utility.hpp"

namespace idrc {

namespace {

Orientation getBoundaryOrient(Rotation rotation, bool is_hole, const PlanarCoord& begin_coord, const PlanarCoord& end_coord)
{
  auto rotate_left = [](Orientation orient) {
    switch (orient) {
      case Orientation::kEast:
        return Orientation::kNorth;
      case Orientation::kNorth:
        return Orientation::kWest;
      case Orientation::kWest:
        return Orientation::kSouth;
      case Orientation::kSouth:
        return Orientation::kEast;
      default:
        return Orientation::kNone;
    }
  };
  auto rotate_right = [](Orientation orient) {
    switch (orient) {
      case Orientation::kEast:
        return Orientation::kSouth;
      case Orientation::kSouth:
        return Orientation::kWest;
      case Orientation::kWest:
        return Orientation::kNorth;
      case Orientation::kNorth:
        return Orientation::kEast;
      default:
        return Orientation::kNone;
    }
  };

  Orientation travel_orient = DRCUTIL.getOrientation(begin_coord, end_coord);
  bool metal_on_left = (rotation == Rotation::kCounterclockwise);
  if (is_hole) {
    metal_on_left = !metal_on_left;
  }
  return metal_on_left ? rotate_right(travel_orient) : rotate_left(travel_orient);
}

void collectBoundaryEdges(GTLHolePolyInt& check_hole_poly, bool is_hole, int32_t polygon_id, std::vector<BoundaryData>& boundary_pool,
                          std::vector<int32_t>& ring_boundary_ids)
{
  int32_t coord_size = static_cast<int32_t>(check_hole_poly.size());
  if (coord_size < 2) {
    return;
  }

  std::vector<PlanarCoord> coord_list;
  coord_list.reserve(coord_size);
  for (auto iter = check_hole_poly.begin(); iter != check_hole_poly.end(); iter++) {
    coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
  }
  if (coord_list.size() < 2) {
    return;
  }

  Rotation rotation = DRCUTIL.getRotation(check_hole_poly);
  std::vector<bool> convex_corner_list(coord_size, false);
  if (coord_size >= 3) {
    for (int32_t i = 0; i < coord_size; i++) {
      PlanarCoord& pre_coord = coord_list[(i - 1 + coord_size) % coord_size];
      PlanarCoord& curr_coord = coord_list[i];
      PlanarCoord& post_coord = coord_list[(i + 1) % coord_size];
      convex_corner_list[i] = is_hole ? DRCUTIL.isConcaveCorner(rotation, pre_coord, curr_coord, post_coord)
                                      : DRCUTIL.isConvexCorner(rotation, pre_coord, curr_coord, post_coord);
    }
  }

  ring_boundary_ids.clear();
  ring_boundary_ids.reserve(coord_size);
  for (int32_t i = 0; i < coord_size; i++) {
    PlanarCoord& pre_coord = coord_list[(i - 1 + coord_size) % coord_size];
    PlanarCoord& curr_coord = coord_list[i];
    if (pre_coord == curr_coord) {
      continue;
    }

    BoundaryData boundary_data;
    boundary_data.edge = DRCUTIL.convertToGTLRectInt(DRCUTIL.getRect(pre_coord, curr_coord));
    boundary_data.begin_coord = pre_coord;
    boundary_data.end_coord = curr_coord;
    boundary_data.orient = getBoundaryOrient(rotation, is_hole, pre_coord, curr_coord);
    boundary_data.polygon_id = polygon_id;
    boundary_data.edge_length = DRCUTIL.getManhattanDistance(pre_coord, curr_coord);
    boundary_data.isConvex = convex_corner_list[i];
    boundary_data.isHole = is_hole;

    boundary_pool.push_back(boundary_data);
    ring_boundary_ids.push_back(static_cast<int32_t>(boundary_pool.size()) - 1);
  }

  int32_t ring_size = static_cast<int32_t>(ring_boundary_ids.size());
  if (ring_size < 2) {
    return;
  }
  for (int32_t i = 0; i < ring_size; i++) {
    BoundaryData& boundary_data = boundary_pool[ring_boundary_ids[i]];
    boundary_data.prev_boundary_id = ring_boundary_ids[(i - 1 + ring_size) % ring_size];
    boundary_data.next_boundary_id = ring_boundary_ids[(i + 1) % ring_size];
  }
}

bool isBoundaryCoveredByEnv(const BoundaryData& boundary_data, const bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>>& env_boundary_rtree,
                            const std::vector<BoundaryData>& env_boundary_pool)
{
  std::vector<std::pair<GTLRectInt, int32_t>> env_boundary_pairs;
  env_boundary_rtree.query(bgi::intersects(boundary_data.edge), std::back_inserter(env_boundary_pairs));
  if (env_boundary_pairs.empty()) {
    return false;
  }

  bool is_horizontal = (boundary_data.begin_coord.get_y() == boundary_data.end_coord.get_y());
  int32_t fixed_coord = is_horizontal ? boundary_data.begin_coord.get_y() : boundary_data.begin_coord.get_x();
  int32_t target_begin = is_horizontal ? std::min(boundary_data.begin_coord.get_x(), boundary_data.end_coord.get_x())
                                       : std::min(boundary_data.begin_coord.get_y(), boundary_data.end_coord.get_y());
  int32_t target_end = is_horizontal ? std::max(boundary_data.begin_coord.get_x(), boundary_data.end_coord.get_x())
                                     : std::max(boundary_data.begin_coord.get_y(), boundary_data.end_coord.get_y());

  std::vector<std::pair<int32_t, int32_t>> covered_ranges;
  covered_ranges.reserve(env_boundary_pairs.size());
  for (const auto& [env_edge, env_boundary_id] : env_boundary_pairs) {
    (void) env_edge;
    const BoundaryData& env_boundary = env_boundary_pool[env_boundary_id];
    if (env_boundary.orient != boundary_data.orient) {
      continue;
    }

    bool env_is_horizontal = (env_boundary.begin_coord.get_y() == env_boundary.end_coord.get_y());
    if (env_is_horizontal != is_horizontal) {
      continue;
    }

    int32_t env_fixed_coord = env_is_horizontal ? env_boundary.begin_coord.get_y() : env_boundary.begin_coord.get_x();
    if (env_fixed_coord != fixed_coord) {
      continue;
    }

    int32_t env_begin = env_is_horizontal ? std::min(env_boundary.begin_coord.get_x(), env_boundary.end_coord.get_x())
                                          : std::min(env_boundary.begin_coord.get_y(), env_boundary.end_coord.get_y());
    int32_t env_end = env_is_horizontal ? std::max(env_boundary.begin_coord.get_x(), env_boundary.end_coord.get_x())
                                        : std::max(env_boundary.begin_coord.get_y(), env_boundary.end_coord.get_y());
    if (env_end < target_begin || target_end < env_begin) {
      continue;
    }
    covered_ranges.emplace_back(std::max(env_begin, target_begin), std::min(env_end, target_end));
  }
  if (covered_ranges.empty()) {
    return false;
  }

  std::sort(covered_ranges.begin(), covered_ranges.end());
  int32_t covered_end = target_begin;
  for (const auto& [range_begin, range_end] : covered_ranges) {
    if (range_begin > covered_end) {
      return false;
    }
    covered_end = std::max(covered_end, range_end);
    if (covered_end >= target_end) {
      return true;
    }
  }
  return false;
}

}  // namespace

// public

void RuleValidator::initInst()
{
  if (_rv_instance == nullptr) {
    _rv_instance = new RuleValidator();
  }
}

RuleValidator& RuleValidator::getInst()
{
  if (_rv_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_rv_instance;
}

void RuleValidator::destroyInst()
{
  if (_rv_instance != nullptr) {
    delete _rv_instance;
    _rv_instance = nullptr;
  }
}

// function
std::vector<Violation> RuleValidator::verify(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list,
                                             std::set<ViolationType>& drc_check_type_set, std::vector<DRCShape>& drc_check_region_list)
{
  auto monitor = Monitor::create();
  DRCLOG.info(Loc::current(), "Starting...");
  RVModel rv_model = initRVModel(drc_env_shape_list, drc_result_shape_list, drc_check_type_set, drc_check_region_list);
  setRVComParam(rv_model);
  buildRVClusterList(rv_model);
  verifyRVModel(rv_model);
  buildViolationList(rv_model);
  // debugPlotRVModel(rv_model, "best");
  DRCLOG.info(Loc::current(), "Completed", monitor ? monitor->getStatsInfo() : "");
  return rv_model.get_violation_list();
}

// private

RuleValidator* RuleValidator::_rv_instance = nullptr;

RVModel RuleValidator::initRVModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list,
                                   std::set<ViolationType>& drc_check_type_set, std::vector<DRCShape>& drc_check_region_list)
{
  RVModel rv_model;
  rv_model.set_drc_env_shape_list(drc_env_shape_list);
  rv_model.set_drc_result_shape_list(drc_result_shape_list);
  rv_model.set_drc_check_type_set(drc_check_type_set);
  rv_model.set_drc_check_region_list(drc_check_region_list);
  return rv_model;
}

void RuleValidator::setRVComParam(RVModel& rv_model)
{
  int32_t only_pitch = DRCDM.getOnlyPitch();
  int32_t cluster_size = 100 * only_pitch;
  int32_t expand_size = 5 * only_pitch;
  /**
   * cluster_size, expand_size
   */
  // clang-format off
  RVComParam rv_com_param(cluster_size, expand_size);
  // clang-format on
  DRCLOG.info(Loc::current(), "cluster_size: ", rv_com_param.get_cluster_size());
  DRCLOG.info(Loc::current(), "expand_size: ", rv_com_param.get_expand_size());
  rv_model.set_rv_com_param(rv_com_param);
}

void RuleValidator::buildRVClusterList(RVModel& rv_model)
{
  std::vector<RVCluster>& rv_cluster_list = rv_model.get_rv_cluster_list();
  int32_t cluster_size = rv_model.get_rv_com_param().get_cluster_size();
  int32_t expand_size = rv_model.get_rv_com_param().get_expand_size();

  PlanarRect bounding_box(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
  int32_t offset_x = -1;
  int32_t offset_y = -1;
  int32_t grid_x_size = -1;
  int32_t grid_y_size = -1;
  {
    for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_env_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_env_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_env_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_env_shape.get_ur_y()));
    }
    for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_result_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_result_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_result_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_result_shape.get_ur_y()));
    }
    offset_x = bounding_box.get_ll_x();
    offset_y = bounding_box.get_ll_y();
    grid_x_size = bounding_box.getXSpan() / cluster_size + 1;
    grid_y_size = bounding_box.getYSpan() / cluster_size + 1;
  }
  rv_cluster_list.resize(grid_x_size * grid_y_size);
  for (int32_t grid_x = 0; grid_x < grid_x_size; grid_x++) {
    for (int32_t grid_y = 0; grid_y < grid_y_size; grid_y++) {
      RVCluster& rv_cluster = rv_cluster_list[grid_x + grid_y * grid_x_size];
      rv_cluster.set_cluster_idx(grid_x + grid_y * grid_x_size);
      rv_cluster.get_cluster_rect_list().emplace_back(grid_x * cluster_size + offset_x, grid_y * cluster_size + offset_y,
                                                      (grid_x + 1) * cluster_size + offset_x, (grid_y + 1) * cluster_size + offset_y);
      rv_cluster.set_rv_com_param(&rv_model.get_rv_com_param());
    }
  }
  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_env_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / cluster_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / cluster_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / cluster_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / cluster_size;
    for (int32_t grid_x = grid_ll_x; grid_x <= grid_ur_x; grid_x++) {
      for (int32_t grid_y = grid_ll_y; grid_y <= grid_ur_y; grid_y++) {
        int32_t cluster_idx = grid_x + grid_y * grid_x_size;
        if (static_cast<int32_t>(rv_cluster_list.size()) <= cluster_idx) {
          DRCLOG.error(Loc::current(), "rv_cluster_list.size() <= cluster_idx!");
        }
        rv_cluster_list[cluster_idx].get_drc_env_shape_list().push_back(&drc_env_shape);
      }
    }
  }
  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_result_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / cluster_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / cluster_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / cluster_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / cluster_size;
    for (int32_t grid_x = grid_ll_x; grid_x <= grid_ur_x; grid_x++) {
      for (int32_t grid_y = grid_ll_y; grid_y <= grid_ur_y; grid_y++) {
        int32_t cluster_idx = grid_x + grid_y * grid_x_size;
        if (static_cast<int32_t>(rv_cluster_list.size()) <= cluster_idx) {
          DRCLOG.error(Loc::current(), "rv_cluster_list.size() <= cluster_idx!");
        }
        rv_cluster_list[cluster_idx].get_drc_result_shape_list().push_back(&drc_result_shape);
      }
    }
  }
  for (RVCluster& rv_cluster : rv_cluster_list) {
    rv_cluster.set_drc_check_type_set(&rv_model.get_drc_check_type_set());
    rv_cluster.set_drc_check_region_list(&rv_model.get_drc_check_region_list());
  }
  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    if (drc_result_shape.get_net_idx() < 0) {
      DRCLOG.error(Loc::current(), "The drc_result_shape_list exist idx < 0!");
    }
  }
}

void RuleValidator::verifyRVModel(RVModel& rv_model)
{
  auto monitor = Monitor::create();
  DRCLOG.info(Loc::current(), "Starting...");
#pragma omp parallel for schedule(dynamic)
  for (RVCluster& rv_cluster : rv_model.get_rv_cluster_list()) {
    buildRVCluster(rv_cluster);
    if (needVerifying(rv_cluster)) {
      buildViolationList(rv_cluster);
    }
  }
  DRCLOG.info(Loc::current(), "Completed", monitor ? monitor->getStatsInfo() : "");
}

void RuleValidator::buildRVCluster(RVCluster& rv_cluster)
{
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();

  std::vector<DRCShape>* drc_check_region_list = rv_cluster.get_drc_check_region_list();
  int32_t expand_size = rv_cluster.get_rv_com_param()->get_expand_size();

  if (!drc_check_region_list->empty()) {
    std::vector<DRCShape*> drc_env_shape_list;
    std::vector<DRCShape*> drc_result_shape_list;
    for (DRCShape& drc_check_region : *drc_check_region_list) {
      PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_check_region.get_rect(), expand_size);
      std::map<bool, std::set<int32_t>> type_layer_idx_map;
      {
        int32_t layer_idx = drc_check_region.get_layer_idx();
        type_layer_idx_map[true].insert({layer_idx - 1, layer_idx, layer_idx + 1});
        std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[layer_idx];
        type_layer_idx_map[false].insert(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
      }
      for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
        if (DRCUTIL.exist(type_layer_idx_map[drc_shape->get_is_routing()], drc_shape->get_layer_idx())
            && DRCUTIL.isClosedOverlap(searched_rect, drc_shape->get_rect())) {
          drc_env_shape_list.push_back(drc_shape);
        }
      }
      for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
        if (DRCUTIL.exist(type_layer_idx_map[drc_shape->get_is_routing()], drc_shape->get_layer_idx())
            && DRCUTIL.isClosedOverlap(searched_rect, drc_shape->get_rect())) {
          drc_result_shape_list.push_back(drc_shape);
        }
      }
    }
    std::sort(drc_env_shape_list.begin(), drc_env_shape_list.end());
    drc_env_shape_list.erase(std::unique(drc_env_shape_list.begin(), drc_env_shape_list.end()), drc_env_shape_list.end());
    std::sort(drc_result_shape_list.begin(), drc_result_shape_list.end());
    drc_result_shape_list.erase(std::unique(drc_result_shape_list.begin(), drc_result_shape_list.end()), drc_result_shape_list.end());
    rv_cluster.set_drc_env_shape_list(drc_env_shape_list);
    rv_cluster.set_drc_result_shape_list(drc_result_shape_list);
  }
}

bool RuleValidator::needVerifying(RVCluster& rv_cluster)
{
  if (rv_cluster.get_drc_result_shape_list().empty()) {
    return false;
  }
  for (DRCShape* drc_result_shape : rv_cluster.get_drc_result_shape_list()) {
    for (PlanarRect& cluster_rect : rv_cluster.get_cluster_rect_list()) {
      if (DRCUTIL.isOpenOverlap(cluster_rect, drc_result_shape->get_rect())) {
        return true;
      }
    }
  }
  return false;
}

void RuleValidator::buildViolationList(RVCluster& rv_cluster)
{
  prepareRVCluster(rv_cluster);
  verifyRVCluster(rv_cluster);

  // destroy cluster cache after verify
  rv_cluster.get_layer_data().clear();

  processRVCluster(rv_cluster);
}

void RuleValidator::prepareRVCluster(RVCluster& rv_cluster)
{
  std::map<int32_t, RVLayerData>& layer_data = rv_cluster.get_layer_data();

  layer_data.clear();
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> env_routing_polysets;

  auto add_shape_to_layer_data = [&](DRCShape* drc_shape, bool is_env_shape) {
    GTLRectInt gtl_rect = DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    if (!drc_shape->get_is_routing()) {
      CutData cut_data;
      cut_data.rect = gtl_rect;
      cut_data.net_idx = drc_shape->get_net_idx();
      cut_data.isEnv = is_env_shape;
      layer_data[drc_shape->get_layer_idx()].cut_pool.push_back(cut_data);
      return;
    }
    layer_data[drc_shape->get_layer_idx()].nets[drc_shape->get_net_idx()].polyset += gtl_rect;
    if (is_env_shape) {
      env_routing_polysets[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += gtl_rect;
    }
  };
  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    add_shape_to_layer_data(drc_shape, true);
  }
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    add_shape_to_layer_data(drc_shape, false);
  }

  for (auto& [layer_idx, rv_layer_data] : layer_data) {
    using DeltaRectRTree = bgi::rtree<GTLRectInt, bgi::quadratic<16>>;

    std::vector<std::pair<GTLRectInt, int32_t>> rect_rtree_inputs;
    std::vector<std::pair<GTLRectInt, int32_t>> boundary_rtree_inputs;
    rv_layer_data.polygon_pool.clear();
    rv_layer_data.max_rect_pool.clear();
    rv_layer_data.boundary_pool.clear();
    for (auto& [net_idx, routing_net] : rv_layer_data.nets) {
      std::vector<BoundaryData> env_boundary_pool;
      bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>> env_boundary_rtree;
      const GTLPolySetInt* env_polyset = nullptr;
      GTLPolySetInt delta_polyset;
      DeltaRectRTree delta_rect_rtree;
      bool has_delta_geometry = false;
      {
        auto layer_env_it = env_routing_polysets.find(layer_idx);
        if (layer_env_it != env_routing_polysets.end()) {
          auto net_env_it = layer_env_it->second.find(net_idx);
          if (net_env_it != layer_env_it->second.end()) {
            env_polyset = &net_env_it->second;
            delta_polyset = routing_net.polyset;
            delta_polyset -= *env_polyset;
            has_delta_geometry = !gtl::empty(delta_polyset);
            if (has_delta_geometry) {
              std::vector<GTLRectInt> delta_rect_list;
              gtl::get_max_rectangles(delta_rect_list, delta_polyset);
              delta_rect_rtree = DeltaRectRTree(delta_rect_list);
            }
          }
        }
      }

      auto build_env_boundary_rtree = [&]() {
        if (env_polyset == nullptr || !has_delta_geometry || !env_boundary_rtree.empty()) {
          return;
        }

        std::vector<std::pair<GTLRectInt, int32_t>> env_boundary_rtree_inputs;
        std::vector<GTLHolePolyInt> env_hole_poly_list;
        env_polyset->get(env_hole_poly_list);
        for (GTLHolePolyInt& env_hole_poly : env_hole_poly_list) {
          std::vector<int32_t> env_ring_boundary_ids;
          collectBoundaryEdges(env_hole_poly, false, -1, env_boundary_pool, env_ring_boundary_ids);
          for (int32_t boundary_id : env_ring_boundary_ids) {
            env_boundary_rtree_inputs.push_back({env_boundary_pool[boundary_id].edge, boundary_id});
          }
          for (auto iter = env_hole_poly.begin_holes(); iter != env_hole_poly.end_holes(); iter++) {
            GTLPolyInt gtl_poly = *iter;
            GTLHolePolyInt env_hole_boundary;
            env_hole_boundary.set(gtl_poly.begin(), gtl_poly.end());
            collectBoundaryEdges(env_hole_boundary, true, -1, env_boundary_pool, env_ring_boundary_ids);
            for (int32_t boundary_id : env_ring_boundary_ids) {
              env_boundary_rtree_inputs.push_back({env_boundary_pool[boundary_id].edge, boundary_id});
            }
          }
        }
        env_boundary_rtree = decltype(env_boundary_rtree)(env_boundary_rtree_inputs);
      };

      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      routing_net.polyset.get(gtl_hole_poly_list);
      routing_net.polygon_begin = static_cast<int32_t>(rv_layer_data.polygon_pool.size());
      routing_net.max_rect_begin = static_cast<int32_t>(rv_layer_data.max_rect_pool.size());
      routing_net.boundary_begin = static_cast<int32_t>(rv_layer_data.boundary_pool.size());
      rv_layer_data.polygon_pool.reserve(rv_layer_data.polygon_pool.size() + gtl_hole_poly_list.size());

      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        int32_t polygon_id = static_cast<int32_t>(rv_layer_data.polygon_pool.size());
        rv_layer_data.polygon_pool.push_back(
            {net_idx, static_cast<int32_t>(rv_layer_data.max_rect_pool.size()), 0, static_cast<int32_t>(rv_layer_data.boundary_pool.size()), 0});
        PolygonData& polygon_data = rv_layer_data.polygon_pool.back();
        polygon_data.hole_poly = gtl_hole_poly;

        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_hole_poly);
        rv_layer_data.max_rect_pool.reserve(rv_layer_data.max_rect_pool.size() + gtl_rect_list.size());
        bool is_polygon_env = (env_polyset != nullptr) && !gtl_rect_list.empty();
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          MaxRectData max_rect_data;
          max_rect_data.rect = gtl_rect;
          max_rect_data.polygon_id = polygon_id;
          if (env_polyset != nullptr) {
            if (!has_delta_geometry) {
              max_rect_data.isEnv = true;
            } else {
              std::vector<GTLRectInt> delta_overlap_list;
              delta_rect_rtree.query(bgi::intersects(gtl_rect), std::back_inserter(delta_overlap_list));
              max_rect_data.isEnv = true;
              for (auto& delta_rect : delta_overlap_list) {
                PlanarRect delta_planar_rect = DRCUTIL.convertToPlanarRect(delta_rect);
                PlanarRect max_rect = DRCUTIL.convertToPlanarRect(max_rect_data.rect);
                if (DRCUTIL.isOpenOverlap(delta_planar_rect, max_rect)) {
                  max_rect_data.isEnv = false;
                  break;
                }
              }
            }
          }
          is_polygon_env = is_polygon_env && max_rect_data.isEnv;

          rv_layer_data.max_rect_pool.push_back(max_rect_data);
          int32_t max_rect_id = static_cast<int32_t>(rv_layer_data.max_rect_pool.size()) - 1;
          rect_rtree_inputs.push_back({gtl_rect, max_rect_id});
        }
        polygon_data.max_rect_count = static_cast<int32_t>(rv_layer_data.max_rect_pool.size()) - polygon_data.max_rect_begin;
        polygon_data.isEnv = is_polygon_env;

        if (!is_polygon_env) {
          build_env_boundary_rtree();
        }
        std::vector<int32_t> ring_boundary_ids;
        collectBoundaryEdges(gtl_hole_poly, false, polygon_id, rv_layer_data.boundary_pool, ring_boundary_ids);
        for (int32_t boundary_id : ring_boundary_ids) {
          BoundaryData& boundary_data = rv_layer_data.boundary_pool[boundary_id];
          if (env_polyset != nullptr) {
            boundary_data.isEnv = is_polygon_env ? true : isBoundaryCoveredByEnv(boundary_data, env_boundary_rtree, env_boundary_pool);
          }
          boundary_rtree_inputs.push_back({boundary_data.edge, boundary_id});
        }
        for (auto iter = gtl_hole_poly.begin_holes(); iter != gtl_hole_poly.end_holes(); iter++) {
          GTLPolyInt gtl_poly = *iter;
          GTLHolePolyInt check_hole_poly;
          check_hole_poly.set(gtl_poly.begin(), gtl_poly.end());
          collectBoundaryEdges(check_hole_poly, true, polygon_id, rv_layer_data.boundary_pool, ring_boundary_ids);
          for (int32_t boundary_id : ring_boundary_ids) {
            BoundaryData& boundary_data = rv_layer_data.boundary_pool[boundary_id];
            if (env_polyset != nullptr) {
              boundary_data.isEnv = is_polygon_env ? true : isBoundaryCoveredByEnv(boundary_data, env_boundary_rtree, env_boundary_pool);
            }
            boundary_rtree_inputs.push_back({boundary_data.edge, boundary_id});
          }
        }
        polygon_data.boundary_count = static_cast<int32_t>(rv_layer_data.boundary_pool.size()) - polygon_data.boundary_begin;
      }
      routing_net.polygon_count = static_cast<int32_t>(rv_layer_data.polygon_pool.size()) - routing_net.polygon_begin;
      routing_net.max_rect_count = static_cast<int32_t>(rv_layer_data.max_rect_pool.size()) - routing_net.max_rect_begin;
      routing_net.boundary_count = static_cast<int32_t>(rv_layer_data.boundary_pool.size()) - routing_net.boundary_begin;
    }

    rv_layer_data.rect_rtrees = decltype(rv_layer_data.rect_rtrees)(rect_rtree_inputs);
    rv_layer_data.boundary_rtrees = decltype(rv_layer_data.boundary_rtrees)(boundary_rtree_inputs);
    rv_layer_data.cut_rtrees = decltype(rv_layer_data.cut_rtrees)(rv_layer_data.cut_pool);
  }
}

void RuleValidator::verifyRVCluster(RVCluster& rv_cluster)
{
  if (needVerifying(rv_cluster, ViolationType::kAdjacentCutSpacing)) {
    verifyAdjacentCutSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kCornerFillSpacing)) {
    verifyCornerFillSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kCornerSpacing)) {
    verifyCornerSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kCutEOLSpacing)) {
    verifyCutEOLSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kCutShort)) {
    verifyCutShort(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kDifferentLayerCutSpacing)) {
    verifyDifferentLayerCutSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kEnclosure)) {
    verifyEnclosure(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kEnclosureEdge)) {
    verifyEnclosureEdge(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kEnclosureParallel)) {
    verifyEnclosureParallel(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kEndOfLineSpacing)) {
    verifyEndOfLineSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kFloatingPatch)) {
    verifyFloatingPatch(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kJogToJogSpacing)) {
    verifyJogToJogSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMaximumWidth)) {
    verifyMaximumWidth(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMaxViaStack)) {
    verifyMaxViaStack(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMetalShort)) {
    verifyMetalShort(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMinHole)) {
    verifyMinHole(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMinimumArea)) {
    verifyMinimumArea(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMinimumCut)) {
    verifyMinimumCut(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMinimumWidth)) {
    verifyMinimumWidth(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kMinStep)) {
    verifyMinStep(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kNonsufficientMetalOverlap)) {
    verifyNonsufficientMetalOverlap(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kNotchSpacing)) {
    verifyNotchSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kOffGridOrWrongWay)) {
    verifyOffGridOrWrongWay(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kOutOfDie)) {
    verifyOutOfDie(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kParallelRunLengthSpacing)) {
    verifyParallelRunLengthSpacing(rv_cluster);
  }
  if (needVerifying(rv_cluster, ViolationType::kSameLayerCutSpacing)) {
    verifySameLayerCutSpacing(rv_cluster);
  }
}

bool RuleValidator::needVerifying(RVCluster& rv_cluster, ViolationType violation_type)
{
  std::set<ViolationType>& exist_rule_set = DRCDM.getDatabase().get_exist_rule_set();

  std::set<ViolationType>* drc_check_type_set = rv_cluster.get_drc_check_type_set();

  if (drc_check_type_set->empty()) {
    return DRCUTIL.exist(exist_rule_set, violation_type);
  } else {
    return (DRCUTIL.exist(*drc_check_type_set, violation_type) && DRCUTIL.exist(exist_rule_set, violation_type));
  }
}

void RuleValidator::processRVCluster(RVCluster& rv_cluster)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : rv_cluster.get_violation_list()) {
    bool has_overlap = false;
    for (PlanarRect& cluster_rect : rv_cluster.get_cluster_rect_list()) {
      if (DRCUTIL.isOpenOverlap(cluster_rect, violation.get_rect())) {
        has_overlap = true;
        break;
      }
    }
    if (!has_overlap) {
      continue;
    }
    new_violation_list.push_back(violation);
  }
  std::sort(new_violation_list.begin(), new_violation_list.end(), CmpViolation());
  new_violation_list.erase(std::unique(new_violation_list.begin(), new_violation_list.end()), new_violation_list.end());
  rv_cluster.set_violation_list(new_violation_list);
}

void RuleValidator::buildViolationList(RVModel& rv_model)
{
  std::vector<Violation>& violation_list = rv_model.get_violation_list();
  for (RVCluster& rv_cluster : rv_model.get_rv_cluster_list()) {
    for (Violation& violation : rv_cluster.get_violation_list()) {
      violation_list.push_back(violation);
    }
  }
  std::sort(violation_list.begin(), violation_list.end(), CmpViolation());
  violation_list.erase(std::unique(violation_list.begin(), violation_list.end()), violation_list.end());
}

#if 1  // aux

int32_t RuleValidator::getIdx(int32_t idx, int32_t coord_size)
{
  return (idx + coord_size) % coord_size;
}

#endif

#if 1  // debug

void RuleValidator::debugPlotRVModel(RVModel& rv_model, std::string flag)
{
  Die& die = DRCDM.getDatabase().get_die();
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(die);
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    GPStruct drc_env_shape_struct(DRCUTIL.getString("drc_env_shape(net_", drc_env_shape.get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kEnvShape));
    gp_boundary.set_rect(drc_env_shape.get_rect());
    if (drc_env_shape.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_env_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_env_shape.get_layer_idx()));
    }
    drc_env_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_env_shape_struct);
  }

  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    GPStruct drc_result_shape_struct(DRCUTIL.getString("drc_result_shape(net_", drc_result_shape.get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kResultShape));
    gp_boundary.set_rect(drc_result_shape.get_rect());
    if (drc_result_shape.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_result_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_result_shape.get_layer_idx()));
    }
    drc_result_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_result_shape_struct);
  }

  for (Violation& violation : rv_model.get_violation_list()) {
    std::string net_idx_name = DRCUTIL.getString("net");
    for (int32_t violation_net_idx : violation.get_violation_net_set()) {
      net_idx_name = DRCUTIL.getString(net_idx_name, ",", violation_net_idx);
    }
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")(rs,", violation.get_required_size(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(DRCGP.convertGPDataType(violation.get_violation_type())));
    gp_boundary.set_rect(violation.get_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(violation.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(violation.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
    gp_gds.addStruct(violation_struct);
  }

  std::string gds_file_path = DRCUTIL.getString(rv_temp_directory_path, flag, "_rv_model.gds");
  DRCGP.plot(gp_gds, gds_file_path);
}

void RuleValidator::debugPlotRVCluster(RVCluster& rv_cluster, std::string flag)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

  GPStruct base_region_struct("base_region");
  for (PlanarRect& cluster_rect : rv_cluster.get_cluster_rect_list()) {
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(cluster_rect);
    base_region_struct.push(gp_boundary);
  }
  gp_gds.addStruct(base_region_struct);

  for (DRCShape* drc_env_shape : rv_cluster.get_drc_env_shape_list()) {
    GPStruct drc_env_shape_struct(DRCUTIL.getString("drc_env_shape(net_", drc_env_shape->get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kEnvShape));
    gp_boundary.set_rect(drc_env_shape->get_rect());
    if (drc_env_shape->get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_env_shape->get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_env_shape->get_layer_idx()));
    }
    drc_env_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_env_shape_struct);
  }

  for (DRCShape* drc_result_shape : rv_cluster.get_drc_result_shape_list()) {
    GPStruct drc_result_shape_struct(DRCUTIL.getString("drc_result_shape(net_", drc_result_shape->get_net_idx(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kResultShape));
    gp_boundary.set_rect(drc_result_shape->get_rect());
    if (drc_result_shape->get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(drc_result_shape->get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(drc_result_shape->get_layer_idx()));
    }
    drc_result_shape_struct.push(gp_boundary);
    gp_gds.addStruct(drc_result_shape_struct);
  }

  for (Violation& violation : rv_cluster.get_violation_list()) {
    std::string net_idx_name = DRCUTIL.getString("net");
    for (int32_t violation_net_idx : violation.get_violation_net_set()) {
      net_idx_name = DRCUTIL.getString(net_idx_name, ",", violation_net_idx);
    }
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")(rs,", violation.get_required_size(), ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(DRCGP.convertGPDataType(violation.get_violation_type())));
    gp_boundary.set_rect(violation.get_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(violation.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(violation.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
    gp_gds.addStruct(violation_struct);
  }

  std::string gds_file_path = DRCUTIL.getString(rv_temp_directory_path, flag, "_rv_cluster_", rv_cluster.get_cluster_idx(), ".gds");

  DRCGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace idrc
