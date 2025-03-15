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
#include "RuleChecker.hpp"

namespace idrc {

// public

void RuleChecker::initInst()
{
  if (_rc_instance == nullptr) {
    _rc_instance = new RuleChecker();
  }
}

RuleChecker& RuleChecker::getInst()
{
  if (_rc_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_rc_instance;
}

void RuleChecker::destroyInst()
{
  if (_rc_instance != nullptr) {
    delete _rc_instance;
    _rc_instance = nullptr;
  }
}

// function

std::vector<Violation> RuleChecker::check(std::vector<DRCShape>& drc_shape_list)
{
  RCModel rc_model = initRCModel(drc_shape_list);
  buildRCModel(rc_model);
  checkRCModel(rc_model);
  return getViolationList(rc_model);
}

// private

RuleChecker* RuleChecker::_rc_instance = nullptr;

RCModel RuleChecker::initRCModel(std::vector<DRCShape>& drc_shape_list)
{
  RCModel rc_model;
  rc_model.set_drc_shape_list(drc_shape_list);
  return rc_model;
}

void RuleChecker::buildRCModel(RCModel& rc_model)
{
  int32_t box_size = 50 * DRCDM.getOnlyPitch();
  int32_t expand_size = 2 * DRCDM.getOnlyPitch();

  PlanarRect bounding_box(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
  int32_t grid_x_size = -1;
  int32_t grid_y_size = -1;
  {
    for (DRCShape& drc_shape : rc_model.get_drc_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_shape.get_ur_y()));
    }
    PlanarRect enlarged_rect = DRCUTIL.getEnlargedRect(bounding_box, 1);
    grid_x_size = std::ceil(enlarged_rect.getXSpan() / 1.0 / box_size);
    grid_y_size = std::ceil(enlarged_rect.getYSpan() / 1.0 / box_size);
  }
  rc_model.get_rc_box_list().resize(grid_x_size * grid_y_size);
  for (DRCShape& drc_shape : rc_model.get_drc_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    for (int32_t grid_x = (searched_rect.get_ll_x() / box_size); grid_x <= (searched_rect.get_ur_x() / box_size); grid_x++) {
      for (int32_t grid_y = (searched_rect.get_ll_y() / box_size); grid_y <= (searched_rect.get_ur_y() / box_size); grid_y++) {
        rc_model.get_rc_box_list()[grid_x + grid_y * grid_x_size].get_drc_shape_list().push_back(&drc_shape);
      }
    }
  }
}

void RuleChecker::checkRCModel(RCModel& rc_model)
{
#pragma omp parallel for
  for (RCBox& rc_box : rc_model.get_rc_box_list()) {
    // preRCBox(rc_box);
    checkRCBox(rc_box);
    // postRCBox(rc_box);
  }
}

void RuleChecker::checkRCBox(RCBox& rc_box)
{
  checkAdjacentCutSpacing(rc_box);
  checkCornerFillSpacing(rc_box);
  checkCutEOLSpacing(rc_box);
  checkCutShort(rc_box);
  checkDifferentLayerCutSpacing(rc_box);
  checkEnclosure(rc_box);
  checkEnclosureEdge(rc_box);
  checkEnclosureParallel(rc_box);
  checkEndOfLineSpacing(rc_box);
  checkFloatingPatch(rc_box);
  checkJogToJogSpacing(rc_box);
  checkMaxViaStack(rc_box);
  checkMetalShort(rc_box);
  checkMinHole(rc_box);
  checkMinimumArea(rc_box);
  checkMinimumCut(rc_box);
  checkMinimumWidth(rc_box);
  checkMinStep(rc_box);
  checkNonsufficientMetalOverlap(rc_box);
  checkNotchSpacing(rc_box);
  checkOffGridOrWrongWay(rc_box);
  checkOutOfDie(rc_box);
  checkParallelRunLengthSpacing(rc_box);
  checkSameLayerCutSpacing(rc_box);
}

std::vector<Violation> RuleChecker::getViolationList(RCModel& rc_model)
{
  std::vector<Violation> violation_list;
  for (RCBox& rc_box : rc_model.get_rc_box_list()) {
    for (Violation& violation : rc_box.get_violation_list()) {
      violation_list.push_back(violation);
    }
  }
  return violation_list;
}

}  // namespace idrc
