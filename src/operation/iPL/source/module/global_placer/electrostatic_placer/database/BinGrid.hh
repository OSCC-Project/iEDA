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

#include <stdint.h>

#include <iostream>
#include <vector>

#include "GridManager.hh"
#include "NesInstance.hh"

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
  ~BinGrid();

  int64_t get_overflow_area_with_filler() const { return _overflow_area_wfiller; }
  int64_t get_overflow_area_without_filler() const { return _overflow_area_wofiller; }

  void set_thread_nums(int32_t thread_nums) { _thread_nums = thread_nums; }

  void initNesInstanceTypeList(std::vector<NesInstance*>& nInst_list);

  void updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num);
  void updataOverflowArea(std::vector<NesInstance*>& nInst_list, int32_t thread_num);

  int64_t obtainOverflowAreaWithoutFiller();
  int64_t obtainOverflowArea();

  std::vector<Grid*> obtainOccupiedBinList(NesInstance* nInst);  // TODO.
  std::vector<NesInstance*> obtainOccupiednInstList(Grid* bin);  // TODO.

 private:
  GridManager* _grid_manager;
  int32_t _thread_nums;

  std::vector<NesInstance*> _macro_inst_list;
  std::vector<NesInstance*> _stdcell_list;
  std::vector<NesInstance*> _filler_list;

  int64_t _overflow_area_wfiller;
  int64_t _overflow_area_wofiller;

  int32_t _bin_cnt_x;
  int32_t _bin_cnt_y;
  int32_t _bin_size_x;
  int32_t _bin_size_y;

  std::vector<std::vector<NesInstance*>> _bin_inst_list;
  std::vector<AreaInfo> _bin_area_list;

  void resetBinToArea();

  void addBinnInstConnection(Grid* bin, NesInstance* nInst);
  void addBinMacroAreaInfo(Grid* bin, int64_t macro_area);
  void addBinStdcellAreaInfo(Grid* bin, int64_t stdcell_area);
  void addBinFillerAreaInfo(Grid* bin, int64_t filler_area);
};
inline BinGrid::BinGrid(GridManager* grid_manager)
    : _grid_manager(grid_manager), _thread_nums(1), _overflow_area_wfiller(INT64_MIN), _overflow_area_wofiller(INT64_MIN)
{
  _bin_cnt_x = _grid_manager->get_grid_size_x();
  _bin_cnt_y = _grid_manager->get_grid_size_y();
  _bin_size_x = _grid_manager->get_grid_size_x();
  _bin_size_y = _grid_manager->get_grid_size_y();

  _bin_inst_list.resize(_bin_cnt_x * _bin_cnt_y);
  _bin_area_list.resize(_bin_cnt_x * _bin_cnt_y);
}

inline BinGrid::~BinGrid()
{
}

inline void BinGrid::resetBinToArea()
{
#pragma omp parallel for num_threads(_thread_nums)
  for (auto& area_info : _bin_area_list) {
    area_info.reset();
  }
}

inline void BinGrid::initNesInstanceTypeList(std::vector<NesInstance*>& nInst_list)
{
  for (auto* nInst : nInst_list) {
    if (nInst->isFixed()) {
      continue;
    }

    if (nInst->isMacro()) {
      _macro_inst_list.push_back(nInst);
    } else if (nInst->isFiller()) {
      _filler_list.push_back(nInst);
    } else {
      _stdcell_list.push_back(nInst);
    }
  }
}

inline void BinGrid::addBinnInstConnection(Grid* bin, NesInstance* nInst)
{
  int32_t grid_index = bin->row_idx * _bin_cnt_x + bin->grid_idx;
  _bin_inst_list[grid_index].push_back(nInst);
}

inline void BinGrid::updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num)
{
  updataOverflowArea(nInst_list, thread_num);

#pragma omp parallel for num_threads(thread_num)
  for (auto* nInst : _filler_list) {
    auto nInst_density_shape = std::move(nInst->get_density_shape());

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, nInst_density_shape);
    for (auto* grid : overlap_grid_list) {
      auto& grid_area_ref = grid->occupied_area;

      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, nInst_density_shape);

      int64_t inst_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());

#pragma omp atomic
      grid_area_ref += inst_area;
    }
  }
}

inline void BinGrid::updataOverflowArea(std::vector<NesInstance*>& nInst_list, int32_t thread_num)
{
  int64_t overflow_area_wofiller = 0;
  _grid_manager->clearAllOccupiedArea();

  for (auto* nInst : _macro_inst_list) {
    auto nInst_density_shape = std::move(nInst->get_density_shape());
    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, nInst_density_shape);

#pragma omp parallel for num_threads(thread_num)
    for (auto* grid : overlap_grid_list) {
      auto& grid_area_ref = grid->occupied_area;

      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, nInst_density_shape);
      int64_t inst_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());

      inst_area *= grid->available_ratio;

#pragma omp atomic
      grid_area_ref += inst_area;
    }
  }

#pragma omp parallel for num_threads(thread_num)
  for (auto* nInst : _stdcell_list) {
    auto nInst_density_shape = std::move(nInst->get_density_shape());

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, nInst_density_shape);
    for (auto* grid : overlap_grid_list) {
      auto& grid_area_ref = grid->occupied_area;

      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, nInst_density_shape);
      int64_t inst_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());

#pragma omp atomic
      grid_area_ref += inst_area;
    }
  }

  for (auto& grid_row : _grid_manager->get_grid_2d_list()) {
    for (auto& grid : grid_row) {
      overflow_area_wofiller += grid.obtainGridOverflowArea();
    }
  }
  _overflow_area_wofiller = overflow_area_wofiller;
}

inline int64_t BinGrid::obtainOverflowAreaWithoutFiller()
{
  int64_t overflow_area = 0;

#pragma omp parallel for num_threads(_thread_nums)
  for (auto& grid_row : _grid_manager->get_grid_2d_list()) {
    for (auto& grid : grid_row) {
      int32_t grid_index = grid.row_idx * _bin_cnt_x + grid.grid_idx;

      auto& area_info = _bin_area_list[grid_index];
      int64_t relative_area = area_info.macro_area + area_info.stdcell_area;

      // bin target area.
      int64_t bin_area = grid.grid_area;
      int64_t target_area = static_cast<int64_t>(bin_area * grid.available_ratio);

#pragma omp atomic
      overflow_area += std::max(int64_t(0), relative_area - target_area);
    }
  }

  return overflow_area;
}

inline int64_t BinGrid::obtainOverflowArea()
{
  int overflow_area = 0;

  for (auto& grid_row : _grid_manager->get_grid_2d_list()) {
    for (auto& grid : grid_row) {
      overflow_area += grid.obtainGridOverflowArea();
    }
  }

  return overflow_area;
}

}  // namespace ipl

#endif
