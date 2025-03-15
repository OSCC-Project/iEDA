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
#include "Module.hpp"

namespace idrc {

// public

void Module::initInst()
{
  if (_mod_instance == nullptr) {
    _mod_instance = new Module();
  }
}

Module& Module::getInst()
{
  if (_mod_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_mod_instance;
}

void Module::destroyInst()
{
  if (_mod_instance != nullptr) {
    delete _mod_instance;
    _mod_instance = nullptr;
  }
}

// function

void Module::check(DRCModel& drc_model)
{
  buildDRCModel(drc_model);
  checkDRCModel(drc_model);
}

// private

Module* Module::_mod_instance = nullptr;

void Module::buildDRCModel(DRCModel& drc_model)
{
  int32_t box_size = 50 * DRCDM.getOnlyPitch();
  int32_t expand_size = 2 * DRCDM.getOnlyPitch();

  PlanarRect bounding_box(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
  int32_t grid_x_size = -1;
  int32_t grid_y_size = -1;
  {
    for (DRCShape& drc_shape : drc_model.get_drc_shape_list()) {
      bounding_box.set_ll_x(std::min(bounding_box.get_ll_x(), drc_shape.get_ll_x()));
      bounding_box.set_ll_y(std::min(bounding_box.get_ll_y(), drc_shape.get_ll_y()));
      bounding_box.set_ur_x(std::max(bounding_box.get_ur_x(), drc_shape.get_ur_x()));
      bounding_box.set_ur_y(std::max(bounding_box.get_ur_y(), drc_shape.get_ur_y()));
    }
    PlanarRect enlarged_rect = DRCUTIL.getEnlargedRect(bounding_box, 1);
    grid_x_size = std::ceil(enlarged_rect.getXSpan() / 1.0 / box_size);
    grid_y_size = std::ceil(enlarged_rect.getYSpan() / 1.0 / box_size);
  }
  drc_model.get_drc_box_list().resize(grid_x_size * grid_y_size);
  for (DRCShape& drc_shape : drc_model.get_drc_shape_list()) {
    PlanarRect searched_rect = DRCUTIL.getEnlargedRect(drc_shape.get_rect(), expand_size);
    searched_rect = DRCUTIL.getRegularRect(searched_rect, bounding_box);
    for (int32_t grid_x = (searched_rect.get_ll_x() / box_size); grid_x <= (searched_rect.get_ur_x() / box_size); grid_x++) {
      for (int32_t grid_y = (searched_rect.get_ll_y() / box_size); grid_y <= (searched_rect.get_ur_y() / box_size); grid_y++) {
        drc_model.get_drc_box_list()[grid_x + grid_y * grid_x_size].get_drc_shape_list().push_back(&drc_shape);
      }
    }
  }
}

void Module::checkDRCModel(DRCModel& drc_model)
{
#pragma omp parallel for
  for (DRCBox& drc_box : drc_model.get_drc_box_list()) {
    // preDRCBox(drc_box);
    checkDRCBox(drc_box);
    // postDRCBox(drc_box);
  }
}

void Module::checkDRCBox(DRCBox& drc_box)
{
  checkAdjacentCutSpacing(drc_box);
  checkCornerFillSpacing(drc_box);
  checkCutEOLSpacing(drc_box);
  checkCutShort(drc_box);
  checkDifferentLayerCutSpacing(drc_box);
  checkEnclosure(drc_box);
  checkEnclosureEdge(drc_box);
  checkEnclosureParallel(drc_box);
  checkEndOfLineSpacing(drc_box);
  checkFloatingPatch(drc_box);
  checkJogToJogSpacing(drc_box);
  checkMaxViaStack(drc_box);
  checkMetalShort(drc_box);
  checkMinHole(drc_box);
  checkMinimumArea(drc_box);
  checkMinimumCut(drc_box);
  checkMinimumWidth(drc_box);
  checkMinStep(drc_box);
  checkNonsufficientMetalOverlap(drc_box);
  checkNotchSpacing(drc_box);
  checkOffGridOrWrongWay(drc_box);
  checkOutOfDie(drc_box);
  checkParallelRunLengthSpacing(drc_box);
  checkSameLayerCutSpacing(drc_box);
}

}  // namespace idrc
