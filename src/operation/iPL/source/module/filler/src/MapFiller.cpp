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
/**
 * @file
 * @author WenruiWu (lixq01@pcl.ac.cn)
 * @brief SubFlow, Filler;
 * @version 0.1
 * @date 2022-4-1
 **/

#include "MapFiller.h"

#include <map>
#include <set>
#include <string>

namespace ipl {

void MapFiller::addFillerCellWithGroups()
{
  for (auto region : _pl_design->get_region_list()) {
    if (region->isFence())
      continue;
    for (auto rect : region->get_boundaries()) {
      std::vector<Rectangle<int32_t>> blockage_list;
      for (Instance* inst : region->get_instances()) {
        if (isInstInside(inst, rect)) {
          int32_t lx = inst->get_coordi().get_x() - _pl_layout->get_core_shape().get_ll_x() - rect.get_ll_x();
          int32_t ly = inst->get_coordi().get_y() - _pl_layout->get_core_shape().get_ll_y() - rect.get_ll_y();
          int32_t ux = inst->get_coordi().get_x() + inst->get_shape_width() - _pl_layout->get_core_shape().get_ll_x() - rect.get_ll_x();
          int32_t uy = inst->get_coordi().get_y() + inst->get_shape_height() - _pl_layout->get_core_shape().get_ll_y() - rect.get_ll_y();
          blockage_list.push_back(Rectangle<int32_t>(lx, ly, ux, uy));
        }
      }
      // Rectangle<int32_t> region_rect(0, 0, rect.get_width(), rect.get_height());
      reset(rect, blockage_list);
      addFillerCell();
    }
  }
}

void MapFiller::addFillerCellWithoutGroups()
{
  std::vector<Rectangle<int32_t>> blockage_list;
  std::set<Instance*> can_cells;
  for (auto inst : _pl_design->get_instance_list()) {
    can_cells.insert(inst);
  }
  for (auto region : _pl_design->get_region_list()) {
    for (Instance* inst : region->get_instances()) {
      can_cells.erase(inst);
    }
  }
  for (Instance* inst : can_cells) {
    int32_t lx = inst->get_coordi().get_x() - _pl_layout->get_core_shape().get_ll_x();
    int32_t ly = inst->get_coordi().get_y() - _pl_layout->get_core_shape().get_ll_y();
    int32_t ux = inst->get_coordi().get_x() + inst->get_shape_width() - _pl_layout->get_core_shape().get_ll_x();
    int32_t uy = inst->get_coordi().get_y() + inst->get_shape_height() - _pl_layout->get_core_shape().get_ll_y();
    blockage_list.push_back(Rectangle<int32_t>(lx, ly, ux, uy));
  }
  for (auto region : _pl_design->get_region_list()) {
    for (Rectangle<int32_t>& rect : region->get_boundaries()) {
      blockage_list.push_back(Rectangle<int32_t>(
          rect.get_ll_x() - _pl_layout->get_core_shape().get_ll_x(), rect.get_ll_y() - _pl_layout->get_core_shape().get_ll_y(),
          rect.get_ur_x() - _pl_layout->get_core_shape().get_ll_x(), rect.get_ur_y() - _pl_layout->get_core_shape().get_ll_y()));
    }
  }
  Rectangle<int32_t> core(0, 0, _pl_layout->get_core_shape().get_width(), _pl_layout->get_core_shape().get_height());
  reset(core, blockage_list);
  addFillerCell();
}

bool MapFiller::isInstInside(Instance* inst, Rectangle<int32_t> rect)
{
  int32_t lx = inst->get_coordi().get_x();
  int32_t ly = inst->get_coordi().get_y();
  int32_t ux = inst->get_coordi().get_x() + inst->get_shape_width();
  int32_t uy = inst->get_coordi().get_y() + inst->get_shape_height();
  return lx >= rect.get_ll_x() && ly >= rect.get_ll_y() && ux >= rect.get_ur_x() && uy >= rect.get_ur_y();
}

void MapFiller::fixed_cell_assign()
{
  std::vector<std::vector<bool>> grid_list(_row_count, std::vector<bool>(_row_site_count, true));
  for (auto& rect : _blockage_list) {
    int32_t x_site_start = rect.get_ll_x() / _site_width;
    int32_t x_site_end = rect.get_ur_x() / _site_width - 1;
    if (rect.get_ur_x() % _site_width != 0) {
      x_site_end += 1;
    }
    int32_t y_row_start = rect.get_ll_y() / _row_height;
    int32_t y_row_end = rect.get_ur_y() / _row_height - 1;
    if (rect.get_ur_y() % _row_height != 0) {
      y_row_end += 1;
    }
    x_site_start = std::max(x_site_start, 0);
    y_row_start = std::max(y_row_start, 0);
    x_site_end = std::min(x_site_end, _row_site_count - 1);
    y_row_end = std::min(y_row_end, _row_count - 1);

    for (int32_t yy = y_row_start; yy <= y_row_end; ++yy) {
      for (int32_t xx = x_site_start; xx <= x_site_end; ++xx) {
        grid_list[yy][xx] = false;
      }
    }
  }
  for (int32_t i = 0; i < _row_count; i++) {
    int32_t l = -1;
    bool last = false;
    for (int32_t j = 0; j < _row_site_count; j++) {
      bool pixel = grid_list[i][j];
      if (pixel) {
        if (!last) {
          l = j;
        }
        last = true;
      } else {
        if (last) {
          _available_sites[i].push_back({l, j - 1});
        }
        last = false;
      }
    }
    if (last) {
      _available_sites[i].push_back({l, _row_site_count - 1});
    }
  }
}

void MapFiller::findFillerMaster(std::vector<std::string> filler_name_list)
{
  _filler_master_list.clear();
  _filler_count.clear();
  for (const auto name : filler_name_list) {
    for (auto* cell : _pl_layout->get_cell_list()) {
      if (name == cell->get_name())
        _filler_master_list.push_back(cell);
    }
  }
  sort(_filler_master_list.begin(), _filler_master_list.end(),
       [&](Cell* cell_a, Cell* cell_b) { return cell_a->get_width() > cell_b->get_width(); });
  _filler_count.resize(_filler_master_list.size());
}

void MapFiller::addFillerCell()
{
  init();
  for (int32_t i = 0; i < _row_count; i++) {
    for (FillerSegment& seg : _available_sites[i]) {
      int32_t& left = seg.l;
      int32_t& right = seg.r;
      while (right - left + 1 >= _min_filler_width) {
        int32_t wspace_width = right - left + 1;
        int32_t flag = 0;
        bool added = false;
        for (auto filler_master : _filler_master_list) {
          int32_t filler_width = filler_master->get_width() / _site_width;
          if (wspace_width - filler_width < _min_filler_width && wspace_width - filler_width != 0)
            continue;
          if (filler_width <= wspace_width) {
            int32_t count = _filler_count[flag]++;
            std::string filler_name = filler_master->get_name() + "_" + std::to_string(count);
            int32_t inst_x = left * _site_width + _region.get_ll_x() + _pl_layout->get_core_shape().get_ll_x();
            int32_t inst_y = i * _row_height + _region.get_ll_y() + _pl_layout->get_core_shape().get_ll_y();
            add_filler_instance(inst_x, inst_y, filler_name, filler_master);
            left += filler_width;
            added = true;
          }
          if (added)
            break;
          flag++;
        }
        if (!added)
          break;
      }
    }
  }
}

void MapFiller::add_filler_instance(int32_t inst_x, int32_t inst_y, std::string filler_name, Cell* filler_master)
{
  Instance* new_inst = new Instance(filler_name);
  new_inst->update_coordi(inst_x, inst_y);
  new_inst->set_cell_master(filler_master);
  Orient orient = _orient_map[inst_y];
  new_inst->set_orient(orient);
  new_inst->set_instance_state(INSTANCE_STATE::kPlaced);
  _pl_design->add_instance(new_inst);
}

// void MapFiller::writefiller_to_ipl()
// {
//   for (auto inst : _filler_inst_list) {
//     auto idb_inst = _idb_design->get_instance_list()->add_instance(inst.get_name());
//     idb_inst->set_cell_master(_master_map_idb[inst.get_master()]);
//     idb_inst->set_orient(idborient_convert(inst.get_orient()));
//     int x = 0, y = 0;
//     inst.get_location(&x, &y);
//     idb_inst->set_coodinate(x, y);
//     for (auto row : _layout.get_rows()) {
//       if (row._y == inst._y)
//         inst.set_orient(row.get_orient());
//     }
//     idb_inst->set_orient(idborient_convert(inst.get_orient()));
//     idb_inst->set_status(IdbPlacementStatus::kPlaced);
//   }
// }

void MapFiller::init()
{
  _available_sites.resize(_row_count);
  fixed_cell_assign();
}

void MapFiller::clear()
{
  for (auto& row_sites : _available_sites) {
    row_sites.clear();
  }
  _blockage_list.clear();
}

void MapFiller::reset(Rectangle<int32_t> region, std::vector<Rectangle<int32_t>> blockage_list)
{
  clear();
  _region = region;
  _blockage_list = std::move(blockage_list);
  _row_count = ceil(1. * _region.get_height() / _row_height);
  _row_site_count = ceil(1. * _region.get_width() / _site_width);
}

void MapFiller::writeBack()
{
}

// main
void MapFiller::mapFillerCell()
{
  for (auto filler_name_list : _filler_group_list) {
    if (filler_name_list.size() == 0)
      break;
    findFillerMaster(filler_name_list);
    if (!_pl_design->get_region_list().empty()) {
      addFillerCellWithGroups();
    }
    addFillerCellWithoutGroups();
  }
}

}  // namespace ipl