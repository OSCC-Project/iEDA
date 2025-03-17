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

#include "GDSPlotter.hpp"

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

std::vector<Violation> RuleValidator::verify(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list)
{
  RVModel rv_model = initRVModel(drc_env_shape_list, drc_result_shape_list);
  // debugPlotRVModel(rv_model, "before");
  setRVComParam(rv_model);
  buildRVModel(rv_model);
  verifyRVModel(rv_model);
  buildViolationList(rv_model);
  // debugPlotRVModel(rv_model, "after");
  return rv_model.get_violation_list();
}

// private

RuleValidator* RuleValidator::_rv_instance = nullptr;

RVModel RuleValidator::initRVModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list)
{
  RVModel rv_model;
  rv_model.set_drc_env_shape_list(drc_env_shape_list);
  rv_model.set_drc_result_shape_list(drc_result_shape_list);
  return rv_model;
}

void RuleValidator::setRVComParam(RVModel& rv_model)
{
  int32_t only_pitch = DRCDM.getOnlyPitch();
  int32_t box_size = 200 * only_pitch;
  int32_t expand_size = 2 * only_pitch;
  /**
   * box_size, expand_size
   */
  // clang-format off
  RVComParam rv_com_param(box_size,expand_size);
  // clang-format on
  DRCLOG.info(Loc::current(), "box_size: ", rv_com_param.get_box_size());
  DRCLOG.info(Loc::current(), "expand_size: ", rv_com_param.get_expand_size());
  rv_model.set_rv_com_param(rv_com_param);
}

void RuleValidator::buildRVModel(RVModel& rv_model)
{
  int32_t box_size = rv_model.get_rv_com_param().get_box_size();
  int32_t expand_size = rv_model.get_rv_com_param().get_expand_size();

  PlanarRect bounding_box(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
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
    PlanarRect enlarged_rect = DRCUTIL.getEnlargedRect(bounding_box, 1);
    grid_x_size = std::ceil(enlarged_rect.getXSpan() / 1.0 / box_size);
    grid_y_size = std::ceil(enlarged_rect.getYSpan() / 1.0 / box_size);
  }
  rv_model.get_rv_box_list().resize(grid_x_size * grid_y_size);
  for (int32_t grid_x = 0; grid_x < grid_x_size; grid_x++) {
    for (int32_t grid_y = 0; grid_y < grid_y_size; grid_y++) {
      RVBox& rv_box = rv_model.get_rv_box_list()[grid_x + grid_y * grid_x_size];
      rv_box.set_box_idx(grid_x + grid_y * grid_x_size);
      rv_box.get_box_rect().set_ll(grid_x * box_size, grid_y * box_size);
      rv_box.get_box_rect().set_ur((grid_x + 1) * box_size, (grid_y + 1) * box_size);
    }
  }
  for (DRCShape& drc_env_shape : rv_model.get_drc_env_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_env_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    for (int32_t grid_x = (searched_rect.get_ll_x() / box_size); grid_x <= (searched_rect.get_ur_x() / box_size); grid_x++) {
      for (int32_t grid_y = (searched_rect.get_ll_y() / box_size); grid_y <= (searched_rect.get_ur_y() / box_size); grid_y++) {
        rv_model.get_rv_box_list()[grid_x + grid_y * grid_x_size].get_drc_env_shape_list().push_back(&drc_env_shape);
      }
    }
  }
  for (DRCShape& drc_result_shape : rv_model.get_drc_result_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_result_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    for (int32_t grid_x = (searched_rect.get_ll_x() / box_size); grid_x <= (searched_rect.get_ur_x() / box_size); grid_x++) {
      for (int32_t grid_y = (searched_rect.get_ll_y() / box_size); grid_y <= (searched_rect.get_ur_y() / box_size); grid_y++) {
        rv_model.get_rv_box_list()[grid_x + grid_y * grid_x_size].get_drc_result_shape_list().push_back(&drc_result_shape);
      }
    }
  }
}

void RuleValidator::verifyRVModel(RVModel& rv_model)
{
#pragma omp parallel for
  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    if (needVerifying(rv_box)) {
      // debugPlotRVBox(rv_box, "before");
      verifyRVBox(rv_box);
      // debugPlotRVBox(rv_box, "middle");
      processRVBox(rv_box);
      // debugPlotRVBox(rv_box, "after");
    }
  }
}

bool RuleValidator::needVerifying(RVBox& rv_box)
{
  if (rv_box.get_drc_result_shape_list().empty()) {
    return false;
  }
  for (DRCShape* drc_result_shape : rv_box.get_drc_result_shape_list()) {
    if (DRCUTIL.isOpenOverlap(rv_box.get_box_rect(), drc_result_shape->get_rect())) {
      return true;
    }
  }
  return false;
}

void RuleValidator::verifyRVBox(RVBox& rv_box)
{
  verifyAdjacentCutSpacing(rv_box);
  verifyCornerFillSpacing(rv_box);
  verifyCutEOLSpacing(rv_box);
  verifyCutShort(rv_box);
  verifyDifferentLayerCutSpacing(rv_box);
  verifyEnclosure(rv_box);
  verifyEnclosureEdge(rv_box);
  verifyEnclosureParallel(rv_box);
  verifyEndOfLineSpacing(rv_box);
  verifyFloatingPatch(rv_box);
  verifyJogToJogSpacing(rv_box);
  verifyMaxViaStack(rv_box);
  verifyMetalShort(rv_box);
  verifyMinHole(rv_box);
  verifyMinimumArea(rv_box);
  verifyMinimumCut(rv_box);
  verifyMinimumWidth(rv_box);
  verifyMinStep(rv_box);
  verifyNonsufficientMetalOverlap(rv_box);
  verifyNotchSpacing(rv_box);
  verifyOffGridOrWrongWay(rv_box);
  verifyOutOfDie(rv_box);
  verifyParallelRunLengthSpacing(rv_box);
  verifySameLayerCutSpacing(rv_box);
}

void RuleValidator::processRVBox(RVBox& rv_box)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : rv_box.get_violation_list()) {
    if (DRCUTIL.isOpenOverlap(rv_box.get_box_rect(), violation.get_rect())) {
      new_violation_list.push_back(violation);
    }
  }
  rv_box.set_violation_list(new_violation_list);
}

void RuleValidator::buildViolationList(RVModel& rv_model)
{
  for (RVBox& rv_box : rv_model.get_rv_box_list()) {
    for (Violation& violation : rv_box.get_violation_list()) {
      rv_model.get_violation_list().push_back(violation);
    }
  }
}

#if 1  // debug

void RuleValidator::debugPlotRVModel(RVModel& rv_model, std::string flag)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

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
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(convertGPDataType(violation.get_violation_type())));
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

void RuleValidator::debugPlotRVBox(RVBox& rv_box, std::string flag)
{
  std::string& rv_temp_directory_path = DRCDM.getConfig().rv_temp_directory_path;

  GPGDS gp_gds;

  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(rv_box.get_box_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  for (DRCShape* drc_env_shape : rv_box.get_drc_env_shape_list()) {
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

  for (DRCShape* drc_result_shape : rv_box.get_drc_result_shape_list()) {
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

  for (Violation& violation : rv_box.get_violation_list()) {
    std::string net_idx_name = DRCUTIL.getString("net");
    for (int32_t violation_net_idx : violation.get_violation_net_set()) {
      net_idx_name = DRCUTIL.getString(net_idx_name, ",", violation_net_idx);
    }
    GPStruct violation_struct(DRCUTIL.getString("violation(", net_idx_name, ")"));
    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(convertGPDataType(violation.get_violation_type())));
    gp_boundary.set_rect(violation.get_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByRouting(violation.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(DRCGP.getGDSIdxByCut(violation.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
    gp_gds.addStruct(violation_struct);
  }

  std::string gds_file_path = DRCUTIL.getString(rv_temp_directory_path, flag, "_rv_box_", rv_box.get_box_idx(), ".gds");

  DRCGP.plot(gp_gds, gds_file_path);
}

GPDataType RuleValidator::convertGPDataType(ViolationType violation_type)
{
  GPDataType gp_data_type;
  switch (violation_type) {
    case ViolationType::kAdjacentCutSpacing:
      gp_data_type = GPDataType::kAdjacentCutSpacing;
      break;
    case ViolationType::kCornerFillSpacing:
      gp_data_type = GPDataType::kCornerFillSpacing;
      break;
    case ViolationType::kCutEOLSpacing:
      gp_data_type = GPDataType::kCutEOLSpacing;
      break;
    case ViolationType::kCutShort:
      gp_data_type = GPDataType::kCutShort;
      break;
    case ViolationType::kDifferentLayerCutSpacing:
      gp_data_type = GPDataType::kDifferentLayerCutSpacing;
      break;
    case ViolationType::kEndOfLineSpacing:
      gp_data_type = GPDataType::kEndOfLineSpacing;
      break;
    case ViolationType::kEnclosure:
      gp_data_type = GPDataType::kEnclosure;
      break;
    case ViolationType::kEnclosureEdge:
      gp_data_type = GPDataType::kEnclosureEdge;
      break;
    case ViolationType::kEnclosureParallel:
      gp_data_type = GPDataType::kEnclosureParallel;
      break;
    case ViolationType::kFloatingPatch:
      gp_data_type = GPDataType::kFloatingPatch;
      break;
    case ViolationType::kJogToJogSpacing:
      gp_data_type = GPDataType::kJogToJogSpacing;
      break;
    case ViolationType::kMaxViaStack:
      gp_data_type = GPDataType::kMaxViaStack;
      break;
    case ViolationType::kMetalShort:
      gp_data_type = GPDataType::kMetalShort;
      break;
    case ViolationType::kMinHole:
      gp_data_type = GPDataType::kMinHole;
      break;
    case ViolationType::kMinimumArea:
      gp_data_type = GPDataType::kMinimumArea;
      break;
    case ViolationType::kMinimumCut:
      gp_data_type = GPDataType::kMinimumCut;
      break;
    case ViolationType::kMinimumWidth:
      gp_data_type = GPDataType::kMinimumWidth;
      break;
    case ViolationType::kMinStep:
      gp_data_type = GPDataType::kMinStep;
      break;
    case ViolationType::kNonsufficientMetalOverlap:
      gp_data_type = GPDataType::kNonsufficientMetalOverlap;
      break;
    case ViolationType::kNotchSpacing:
      gp_data_type = GPDataType::kNotchSpacing;
      break;
    case ViolationType::kOffGridOrWrongWay:
      gp_data_type = GPDataType::kOffGridOrWrongWay;
      break;
    case ViolationType::kOutOfDie:
      gp_data_type = GPDataType::kOutOfDie;
      break;
    case ViolationType::kParallelRunLengthSpacing:
      gp_data_type = GPDataType::kParallelRunLengthSpacing;
      break;
    case ViolationType::kSameLayerCutSpacing:
      gp_data_type = GPDataType::kSameLayerCutSpacing;
      break;
    default:
      DRCLOG.error(Loc::current(), "The violation_type not support!");
      break;
  }
  return gp_data_type;
}

#endif

}  // namespace idrc
