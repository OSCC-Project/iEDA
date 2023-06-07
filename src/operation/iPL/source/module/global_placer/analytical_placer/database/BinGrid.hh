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
 * @Date: 2022-03-09 21:32:52
 * @LastEditTime: 2022-10-27 19:33:55
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/database/BinGrid.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_NESTEROV_PLACE_DATABASE_BIN_GRID_H
#define IPL_OPERATOR_NESTEROV_PLACE_DATABASE_BIN_GRID_H

#include <vector>

#include "GridManager.hh"
#include "NesInstance.hh"
#include "omp.h"

namespace ipl {

struct AreaInfo
{
  AreaInfo();

  void reset();

  int64_t macro_area;
  int64_t stdcell_area;
  int64_t filler_area;
};
inline AreaInfo::AreaInfo() : macro_area(int64_t(0)), stdcell_area(int64_t(0)), filler_area(int64_t(0))
{
}
inline void AreaInfo::reset()
{
  macro_area = 0;
  stdcell_area = 0;
  filler_area = 0;
}

class BinGrid
{
 public:
  BinGrid() = delete;
  explicit BinGrid(GridManager* grid_manager);
  BinGrid(const BinGrid&) = delete;
  BinGrid(BinGrid&&) = delete;
  ~BinGrid() = default;

  void updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num);
  int64_t obtainOverflowAreaWithoutFiller();

  std::vector<Grid*> obtainOccupiedBinList(NesInstance* nInst);  // TODO.
  std::vector<NesInstance*> obtainOccupiednInstList(Grid* bin);  // TODO.

 private:
  GridManager* _grid_manager;

  int32_t _bin_cnt_x;
  int32_t _bin_cnt_y;

  std::vector<std::vector<NesInstance*>> _bin_inst_list;
  std::vector<AreaInfo> _bin_area_list;

  void resetBinToArea();

  void addBinnInstConnection(Grid* bin, NesInstance* nInst);
  void addBinMacroAreaInfo(Grid* bin, int64_t macro_area);
  void addBinStdcellAreaInfo(Grid* bin, int64_t stdcell_area);
  void addBinFillerAreaInfo(Grid* bin, int64_t filler_area);
};
inline BinGrid::BinGrid(GridManager* grid_manager) : _grid_manager(grid_manager)
{
  _bin_cnt_x = _grid_manager->get_grid_size_x();
  _bin_cnt_y = _grid_manager->get_grid_size_y();

  _bin_inst_list.resize(_bin_cnt_x * _bin_cnt_y);
  _bin_area_list.resize(_bin_cnt_x * _bin_cnt_y);
}

inline void BinGrid::resetBinToArea()
{
  for (auto& area_info : _bin_area_list) {
    area_info.reset();
  }
}

inline void BinGrid::addBinnInstConnection(Grid* bin, NesInstance* nInst)
{
  int32_t grid_index = bin->get_row_idx() * _bin_cnt_x + bin->get_grid_idx();
  _bin_inst_list[grid_index].push_back(nInst);
}

inline void BinGrid::updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num)
{
  _grid_manager->clearAllOccupiedArea();
  resetBinToArea();

  for (auto* nInst : nInst_list) {
    if (nInst->isFixed()) {
      continue;
    }

    auto nInst_density_shape = std::move(nInst->get_density_shape());

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, nInst_density_shape);

    for (auto* grid : overlap_grid_list) {
      // addBinnInstConnection(grid, nInst);
      int32_t grid_index = grid->get_row_idx() * _bin_cnt_x + grid->get_grid_idx();

      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, nInst_density_shape);
      if (nInst->isMacro()) {
        int64_t macro_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale() * grid->get_available_ratio());
        _bin_area_list[grid_index].macro_area += macro_area;
        grid->add_area(macro_area);
      } else if (nInst->isFiller()) {
        int64_t filler_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());
        _bin_area_list[grid_index].filler_area += filler_area;
        grid->add_area(filler_area);
      } else {
        int64_t stdcell_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());
        _bin_area_list[grid_index].stdcell_area += stdcell_area;
        grid->add_area(stdcell_area);
      }
    }
  }
}

inline int64_t BinGrid::obtainOverflowAreaWithoutFiller()
{
  int64_t overflow_area = 0;
  for (auto* row : _grid_manager->get_row_list()) {
    int32_t row_index = row->get_row_idx();
    for (auto* grid : row->get_grid_list()) {
      int32_t grid_index = row_index * _bin_cnt_x + grid->get_grid_idx();

      auto& area_info = _bin_area_list[grid_index];
      int64_t relative_area = area_info.macro_area + area_info.stdcell_area;

      // bin target area.
      int64_t bin_area = static_cast<int64_t>(grid->get_width()) * static_cast<int64_t>(grid->get_height());
      int64_t target_area = static_cast<int64_t>(bin_area * grid->get_available_ratio());

      overflow_area += std::max(int64_t(0), relative_area - target_area);
    }
  }

  return overflow_area;
}

}  // namespace ipl

#endif
