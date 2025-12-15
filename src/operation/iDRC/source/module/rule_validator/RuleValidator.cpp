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

namespace idrc {

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
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
  RVModel rv_model = initRVModel(drc_env_shape_list, drc_result_shape_list, drc_check_type_set, drc_check_region_list);
  setRVComParam(rv_model);
  buildRVClusterList(rv_model);
  verifyRVModel(rv_model);
  buildViolationList(rv_model);
  // debugPlotRVModel(rv_model, "best");
  // debugOutputViolation(rv_model);
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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
  int32_t grid_size = 100 * only_pitch;
  int32_t expand_size = 5 * only_pitch;
  /**
   * grid_size, expand_size
   */
  // clang-format off
  RVComParam rv_com_param(grid_size, expand_size);
  // clang-format on
  DRCLOG.info(Loc::current(), "grid_size: ", rv_com_param.get_grid_size());
  DRCLOG.info(Loc::current(), "expand_size: ", rv_com_param.get_expand_size());
  rv_model.set_rv_com_param(rv_com_param);
}

void RuleValidator::buildRVClusterList(RVModel& rv_model)
{
  std::vector<RVCluster>& rv_cluster_list = rv_model.get_rv_cluster_list();
  int32_t grid_size = rv_model.get_rv_com_param().get_grid_size();
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
    grid_x_size = bounding_box.getXSpan() / grid_size + 1;
    grid_y_size = bounding_box.getYSpan() / grid_size + 1;
  }
  rv_cluster_list.resize(grid_x_size * grid_y_size);
  for (int32_t grid_x = 0; grid_x < grid_x_size; grid_x++) {
    for (int32_t grid_y = 0; grid_y < grid_y_size; grid_y++) {
      RVCluster& rv_cluster = rv_cluster_list[grid_x + grid_y * grid_x_size];
      rv_cluster.set_cluster_idx(grid_x + grid_y * grid_x_size);
      rv_cluster.get_cluster_rect_list().emplace_back(grid_x * grid_size + offset_x, grid_y * grid_size + offset_y, (grid_x + 1) * grid_size + offset_x,
                                                      (grid_y + 1) * grid_size + offset_y);
      rv_cluster.set_rv_com_param(&rv_model.get_rv_com_param());
    }
  }
  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_env_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / grid_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / grid_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / grid_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / grid_size;
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
    int32_t grid_ll_x = (searched_rect.get_ll_x() - offset_x) / grid_size;
    int32_t grid_ll_y = (searched_rect.get_ll_y() - offset_y) / grid_size;
    int32_t grid_ur_x = (searched_rect.get_ur_x() - offset_x) / grid_size;
    int32_t grid_ur_y = (searched_rect.get_ur_y() - offset_y) / grid_size;
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
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");
#pragma omp parallel for
  for (RVCluster& rv_cluster : rv_model.get_rv_cluster_list()) {
    buildRVCluster(rv_cluster);
    if (needVerifying(rv_cluster)) {
      buildEnvViolation(rv_cluster);
      buildViolationList(rv_cluster);
      // debugPlotRVCluster(rv_cluster, "best");
    }
  }
  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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

void RuleValidator::buildEnvViolation(RVCluster& rv_cluster)
{
  std::vector<DRCShape*> temp = rv_cluster.get_drc_result_shape_list();
  rv_cluster.get_drc_result_shape_list().clear();
  verifyRVCluster(rv_cluster);
  processRVCluster(rv_cluster);
  for (Violation& violation : rv_cluster.get_violation_list()) {
    rv_cluster.get_env_violation_set().insert(violation);
  }
  rv_cluster.set_drc_result_shape_list(temp);
  rv_cluster.get_violation_list().clear();
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
    if (DRCUTIL.exist(rv_cluster.get_env_violation_set(), violation)) {
      continue;
    }
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

void RuleValidator::buildViolationList(RVCluster& rv_cluster)
{
  verifyRVCluster(rv_cluster);
  processRVCluster(rv_cluster);
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

void RuleValidator::debugOutputViolation(RVModel& rv_model)
{
  Monitor monitor;
  DRCLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  std::map<ViolationType, std::vector<Violation*>> type_violation_list_map;
  for (Violation& violation : rv_model.get_violation_list()) {
    type_violation_list_map[violation.get_violation_type()].push_back(&violation);
  }
  for (auto& [type, violation_list] : type_violation_list_map) {
    std::ofstream* violation_file = DRCUTIL.getOutputFileStream(DRCUTIL.getString(rv_temp_directory_path, GetViolationTypeName()(type), ".txt"));
    for (Violation* violation : violation_list) {
      DRCUTIL.pushStream(violation_file, violation->get_ll_x(), " ", violation->get_ll_y(), " ", violation->get_ur_x(), " ", violation->get_ur_y(), " ");
      if (violation->get_is_routing()) {
        DRCUTIL.pushStream(violation_file, routing_layer_list[violation->get_layer_idx()].get_layer_name(), " ");
      } else {
        DRCUTIL.pushStream(violation_file, cut_layer_list[violation->get_layer_idx()].get_layer_name(), " ");
      }
      DRCUTIL.pushStream(violation_file, violation->get_is_routing() ? "true" : "false", " ");

      DRCUTIL.pushStream(violation_file, "{ ");
      for (int32_t net_idx : violation->get_violation_net_set()) {
        DRCUTIL.pushStream(violation_file, net_idx, " ");
      }
      DRCUTIL.pushStream(violation_file, "}", " ");

      DRCUTIL.pushStream(violation_file, violation->get_required_size(), " ");
      DRCUTIL.pushStream(violation_file, "\n");
    }
    DRCUTIL.closeFileStream(violation_file);
  }

  DRCLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

#endif

}  // namespace idrc
