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
 * @Date: 2022-02-28 14:57:54
 * @LastEditTime: 2022-10-31 22:36:44
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/grid_manager/GridManager.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_UTIL_GRID_MANAGER_H
#define IPL_UTIL_GRID_MANAGER_H

#include <string>
#include <vector>

#include "data/Rectangle.hh"
#include "module/logger/Log.hh"
#include "utility/Utility.hh"

namespace ipl {

class Grid
{
 public:
  Grid() = delete;
  Grid(int32_t row_idx, int32_t grid_idx);
  Grid(const Grid&) = delete;
  Grid(Grid&&) = delete;
  ~Grid() = default;

  Grid& operator=(const Grid&) = delete;
  Grid& operator=(Grid&&) = delete;

  // getter.
  int32_t get_row_idx() const { return _row_idx; }
  int32_t get_grid_idx() const { return _grid_idx; }

  int32_t get_width() const { return _width; }
  int32_t get_height() const { return _height; }
  Rectangle<int32_t> get_shape() const { return _shape; }

  float get_available_ratio() const { return _available_ratio; }
  int64_t get_occupied_area() const { return _occupied_area; }
  int64_t& get_occupied_area_ref() {return _occupied_area;}
  int64_t get_fixed_area() const { return _fixed_area; }

  // setter.
  void set_shape(Rectangle<int32_t> shape);
  void set_available_ratio(float ratio) { _available_ratio = ratio; }
  void add_area(int64_t area) { _occupied_area += area; }
  void add_fixed_area(int64_t area) { _fixed_area += area; }

  // function.
  int64_t obtainAvailableArea();
  void clearOccupiedArea();
  void clearFixedArea();
  void regainFreeArea(int64_t free_area);
  float obtainGridDensity();
  int64_t obtainGridOverflowArea();

 private:
  int32_t _row_idx;
  int32_t _grid_idx;

  int32_t _width;
  int32_t _height;
  Rectangle<int32_t> _shape;

  float _available_ratio;
  int64_t _occupied_area;
  int64_t _fixed_area;
};

inline Grid::Grid(int32_t row_idx, int32_t grid_idx)
    : _row_idx(row_idx), _grid_idx(grid_idx), _available_ratio(1.0), _occupied_area(0), _fixed_area(0)
{
}

class GridRow
{
 public:
  GridRow() = delete;
  GridRow(int32_t row_idx);
  GridRow(const GridRow&) = delete;
  GridRow(GridRow&&) = delete;
  ~GridRow();

  GridRow& operator=(const GridRow&) = delete;
  GridRow& operator=(GridRow&&) = delete;

  // getter.
  int32_t get_row_idx() const { return _row_idx; }
  Rectangle<int32_t> get_shape() const { return _shape; }
  const std::vector<Grid*>& get_grid_list() { return _grid_list; }

  // setter.
  void set_shape(Rectangle<int32_t> shape) { _shape = std::move(shape); }
  void add_grid_list(Grid* grid) { _grid_list.push_back(grid); }

  // function.
  void obtainOverlapGridList(std::vector<Grid*>& grid_list, int32_t lower_x, int32_t upper_x);

  std::vector<Grid*> obtainAvailableGridList(float available_ratio);
  std::vector<Grid*> obtainAvailableGridList(int32_t grid_left, int32_t grid_right, float available_ratio);
  std::vector<Rectangle<int32_t>> calContinuouslyShapeList(std::vector<Grid*> grid_list);
  std::vector<Rectangle<int32_t>> obtainAvailableGridShapeList(float available_ratio);
  void clearAllOccupiedArea();

 private:
  Utility _utility;

  int32_t _row_idx;
  Rectangle<int32_t> _shape;
  std::vector<Grid*> _grid_list;
};

inline GridRow::GridRow(int32_t row_idx) : _utility(Utility()), _row_idx(row_idx)
{
}

inline GridRow::~GridRow()
{
  for (auto* grid : _grid_list) {
    delete grid;
  }
  _grid_list.clear();
}

class GridManager
{
 public:
  GridManager() = delete;
  GridManager(Rectangle<int32_t> region, int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio);
  GridManager(const GridManager&) = delete;
  GridManager(GridManager&&) = delete;
  ~GridManager();

  GridManager& operator=(const GridManager&) = delete;
  GridManager& operator=(GridManager&&) = delete;

  // getter.
  Rectangle<int32_t> get_shape() const { return _shape; }
  const std::vector<GridRow*>& get_row_list() { return _row_list; }
  int32_t get_grid_size_x() const { return _grid_size_x; }
  int32_t get_grid_size_y() const { return _grid_size_y; }

  // setter.
  void add_row_list(GridRow* row) { _row_list.push_back(row); }

  // function.
  int32_t obtainGridCntX();
  int32_t obtainRowCntY();
  void obtainOverlapRowList(std::vector<GridRow*>& row_list, int32_t lower_y, int32_t upper_y);
  void obtainOverlapGridList(std::vector<Grid*>& grid_list, Rectangle<int32_t>& rect);
  std::vector<Rectangle<int32_t>> obtainAvailableRectList(int32_t row_low, int32_t row_high, int32_t grid_left, int32_t grid_right,
                                                          float available_ratio);

  std::vector<Grid*> obtainOverflowIllegalGridList();
  void clearAllOccupiedArea();

  int64_t obtainOverlapArea(Grid* grid, const Rectangle<int32_t>& rect);

 private:
  Utility _utility;

  int32_t _grid_size_x;
  int32_t _grid_size_y;

  Rectangle<int32_t> _shape;
  std::vector<GridRow*> _row_list;

  void init(int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio);
};

inline GridManager::GridManager(Rectangle<int32_t> region_size, int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio)
    : _utility(Utility()), _shape(std::move(region_size))
{
  init(grid_cnt_x, grid_cnt_y, available_ratio);
}

inline GridManager::~GridManager()
{
  for (auto* row : _row_list) {
    delete row;
  }
  _row_list.clear();
}

inline int64_t GridManager::obtainOverlapArea(Grid* grid, const Rectangle<int32_t>& rect)
{
  auto grid_shape = grid->get_shape();

  int64_t overlap_rect_lx = std::max(grid_shape.get_ll_x(), rect.get_ll_x());
  int64_t overlap_rect_ly = std::max(grid_shape.get_ll_y(), rect.get_ll_y());
  int64_t overlap_rect_ux = std::min(grid_shape.get_ur_x(), rect.get_ur_x());
  int64_t overlap_rect_uy = std::min(grid_shape.get_ur_y(), rect.get_ur_y());

  if (overlap_rect_lx >= overlap_rect_ux || overlap_rect_ly >= overlap_rect_uy) {
    LOG_WARNING << "Overlap of grid and input rect produce wrong rectangle!";
    return 0;
  } else {
    return (overlap_rect_ux - overlap_rect_lx) * (overlap_rect_uy - overlap_rect_ly);
  }
}

}  // namespace ipl

#endif