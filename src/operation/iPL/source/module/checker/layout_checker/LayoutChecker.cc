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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-02 15:23:58
 * @LastEditTime: 2022-12-01 11:54:12
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/checker/LayoutChecker.cc
 * Contact : https://github.com/sjchanson
 */

#include "LayoutChecker.hh"

#include <set>

namespace ipl {

LayoutChecker::LayoutChecker(PlacerDB* placer_db)
{
  _placer_db = placer_db;
  _grid_manager = placer_db->get_grid_manager();
  _row_height = placer_db->get_layout()->get_row_height();
  _site_width = placer_db->get_layout()->get_site_width();
  Rectangle<int32_t> core_shape = placer_db->get_layout()->get_core_shape();
  _core_x_range = std::make_pair(core_shape.get_ll_x(), core_shape.get_ur_x());
  _core_y_range = std::make_pair(core_shape.get_ll_y(), core_shape.get_ur_y());
  _inst_to_sites.clear();
  _site_to_insts.clear();
}

bool LayoutChecker::checkInsideCore(Rectangle<int32_t> shape)
{
  return (shape.get_ll_x() >= _core_x_range.first) && (shape.get_ur_x() <= _core_x_range.second)
         && (shape.get_ll_y() >= _core_y_range.first) && (shape.get_ur_y() <= _core_y_range.second);
}

bool LayoutChecker::checkAlignRowSite(Rectangle<int32_t> shape)
{
  Point<int32_t> core_lower = _placer_db->get_layout()->get_core_shape().get_lower_left();

  if ((shape.get_ll_x() - core_lower.get_x()) % _site_width != 0 || (shape.get_ur_x() - core_lower.get_x()) % _site_width != 0) {
    return false;
  }

  if ((shape.get_ll_y() - core_lower.get_y()) % _row_height != 0 || (shape.get_ur_y() - core_lower.get_y()) % _row_height != 0) {
    return false;
  }
  return true;
}

bool LayoutChecker::checkAlignPower(Instance* inst)
{
  int32_t inst_lly = inst->get_shape().get_ll_y();
  int32_t core_lly = _core_y_range.first;

  // make sure the inst is allign in row
  if ((inst_lly - core_lly) % _row_height == 0) {
    int32_t row_id = (inst_lly - core_lly) / _row_height;
    // auto* row = _placer_db->get_layout()->find_row(row_id);
    Orient row_orient = _placer_db->get_layout()->find_row_orient(row_id);

    if (inst->get_orient() != row_orient) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

bool LayoutChecker::isAllPlacedInstInsideCore()
{
  bool is_legal = true;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }

    if (!checkInsideCore(inst->get_shape())) {
      is_legal = false;
      break;
    }
  }
  return is_legal;
}

bool LayoutChecker::isAllPlacedInstAlignRowSite()
{
  bool is_legal = true;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }
    // ignore macros
    if (inst->get_cell_master()->isMacro()) {
      continue;
    }

    if (!checkAlignRowSite(inst->get_shape())) {
      is_legal = false;
      break;
    }
  }
  return is_legal;
}

bool LayoutChecker::isAllPlacedInstAlignPower()
{
  bool is_legal = true;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }
    // ignore macros
    if (inst->get_cell_master()->isMacro()) {
      continue;
    }
    if (!checkAlignPower(inst)) {
      is_legal = false;
      break;
    }
  }
  return is_legal;
}

bool LayoutChecker::isNoOverlapAmongInsts()
{
  std::vector<Grid*> overlap_site_list;
  _grid_manager->obtainOverflowIllegalGridList(overlap_site_list);

  if (!overlap_site_list.empty()) {
    return false;
  } else {
    return true;
  }
}

std::vector<Instance*> LayoutChecker::obtainIllegalInstInsideCore()
{
  std::vector<Instance*> illegal_inst_list;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }

    if (!checkInsideCore(inst->get_shape())) {
      illegal_inst_list.push_back(inst);
    }
  }
  return illegal_inst_list;
}

std::vector<Instance*> LayoutChecker::obtainIllegalInstAlignRowSite()
{
  std::vector<Instance*> illegal_inst_list;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }
    // ignore macros
    if (inst->get_cell_master() && inst->get_cell_master()->isMacro()) {
      continue;
    }

    if (!checkAlignRowSite(inst->get_shape())) {
      illegal_inst_list.push_back(inst);
    }
  }
  return illegal_inst_list;
}

std::vector<Instance*> LayoutChecker::obtainIllegalInstAlignPower()
{
  std::vector<Instance*> illegal_inst_list;
  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    // ignore fake insts.
    if (inst->isFakeInstance()) {
      continue;
    }
    // ignore macros
    if (inst->get_cell_master()->isMacro()) {
      continue;
    }
    // ignore fixed insts.
    if (inst->isFixed()) {
      continue;
    }

    if (!checkAlignPower(inst)) {
      illegal_inst_list.push_back(inst);
    }
  }
  return illegal_inst_list;
}

std::vector<std::vector<Instance*>> LayoutChecker::obtainOverlapInstClique()
{
  updateSiteInstConnection();

  std::vector<std::vector<Instance*>> clique_list;
  std::vector<Grid*> overflow_site_list;
  _grid_manager->obtainOverflowIllegalGridList(overflow_site_list);
  for (auto* site : overflow_site_list) {
    clique_list.push_back(obtainOccupiedInstList(site));
  }

  return clique_list;
}

std::vector<Rectangle<int32_t>> LayoutChecker::obtainWhiteSiteList()
{
  std::vector<Rectangle<int32_t>> white_site_list;

  auto& grid_2d_list = _grid_manager->get_grid_2d_list();
  for (auto& grid_row : grid_2d_list) {
    for (auto& grid : grid_row) {
      if (grid.occupied_area + grid.fixed_area == 0) {
        white_site_list.push_back(grid.shape);
      }
    }
  }

  // for (auto* grid_row : _grid_manager->get_row_list()) {
  //   for (auto* grid : grid_row->get_grid_list()) {
  //     if ((grid->get_occupied_area() + grid->get_fixed_area()) == 0) {
  //       white_site_list.push_back(grid->get_shape());
  //     }
  //   }
  // }
  return white_site_list;
}

void LayoutChecker::updateSiteInstConnection()
{
  clearSiteInstConnection();

  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    if (inst->isOutsideInstance()) {
      continue;
    }
    connectInstSite(inst);
  }
}

void LayoutChecker::connectInstSite(Instance* inst)
{
  // add inst position.
  std::vector<Grid*> overlap_site_list;
  auto inst_shape = std::move(inst->get_shape());
  _grid_manager->obtainOverlapGridList(overlap_site_list, inst_shape);

  for (auto* site : overlap_site_list) {
    addSiteInstConnection(site, inst);
  }
}

void LayoutChecker::addSiteInstConnection(Grid* site, Instance* inst)
{
  _inst_to_sites.emplace(inst, site);
  _site_to_insts.emplace(site, inst);
}

void LayoutChecker::clearSiteInstConnection()
{
  _inst_to_sites.clear();
  _site_to_insts.clear();
}

std::vector<Grid*> LayoutChecker::obtainOccupiedSiteList(Instance* inst)
{
  std::vector<Grid*> site_list;

  int num = _inst_to_sites.count(inst);
  if (num == 0) {
    // do nothing.
  } else if (num == 1) {
    site_list.push_back(_inst_to_sites.find(inst)->second);
  } else {
    auto site_iter = _inst_to_sites.equal_range(inst);
    while (site_iter.first != site_iter.second) {
      site_list.push_back(site_iter.first->second);
      ++site_iter.first;
    }
  }

  return site_list;
}

std::vector<Instance*> LayoutChecker::obtainOccupiedInstList(Grid* site)
{
  std::vector<Instance*> inst_list;

  int num = _site_to_insts.count(site);
  if (num == 0) {
    // do nothing.
  } else if (num == 1) {
    inst_list.push_back(_site_to_insts.find(site)->second);
  } else {
    auto inst_iter = _site_to_insts.equal_range(site);
    while (inst_iter.first != inst_iter.second) {
      inst_list.push_back(inst_iter.first->second);
      ++inst_iter.first;
    }
  }

  return inst_list;
}

// Orient LayoutChecker::obtainLayoutRowOrient(GridRow* grid_row)
// {
//   int32_t row_ll_y = grid_row->get_shape().get_ll_y();
//   int32_t row_ur_y = grid_row->get_shape().get_ur_y();

//   auto layout_row_list = _placer_db->get_layout()->get_row_list();

//   Orient row_orient = Orient::kN_R0;
//   for (auto* layout_row : layout_row_list) {
//     if (layout_row->get_shape().get_ll_y() == row_ll_y && layout_row->get_shape().get_ur_y() == row_ur_y) {
//       row_orient = layout_row->get_orient();
//       break;
//     }
//   }

//   return row_orient;
// }

}  // namespace ipl