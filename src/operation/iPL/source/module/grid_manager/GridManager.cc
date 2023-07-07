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

#include "GridManager.hh"

#include "omp.h"

namespace ipl {

int64_t Grid::obtainAvailableArea()
{
  return static_cast<int64_t>(this->grid_area * this->available_ratio) - this->occupied_area - this->fixed_area;
}

int64_t Grid::obtainGridOverflowArea()
{
  return std::max(int64_t(0), static_cast<int64_t>(this->occupied_area + this->fixed_area - (this->available_ratio * this->grid_area)));
}

float Grid::obtainGridDensity()
{
  return static_cast<float>(this->occupied_area + this->fixed_area) / this->grid_area;
}

void GridManager::obtainOverlapGridList(std::vector<Grid*>& grid_list, Rectangle<int32_t>& rect)
{
  LOG_ERROR_IF(!grid_list.empty()) << "Pass overlap_grid_list is not Empty!";

  std::pair<int, int> y_range = _utility.obtainMinMaxIdx(_shape.get_ll_y(), _grid_size_y, rect.get_ll_y(), rect.get_ur_y());
  std::pair<int, int> x_range = _utility.obtainMinMaxIdx(_shape.get_ll_x(), _grid_size_x, rect.get_ll_x(), rect.get_ur_x());

  int32_t y_cnt = y_range.second - y_range.first;
  int32_t x_cnt = x_range.second - x_range.first;
  grid_list.resize(y_cnt * x_cnt);

#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = y_range.first; i < y_range.second; i++) {
    for (int32_t j = x_range.first; j < x_range.second; j++) {
      grid_list[(i - y_range.first) * x_cnt + (j - x_range.first)] = &_grid_2d_list[i][j];
    }
  }
}

std::vector<Rectangle<int32_t>> GridManager::obtainAvailableRectList(int32_t row_low, int32_t row_high, int32_t grid_left,
                                                                     int32_t grid_right, float available_ratio)
{
  std::vector<Rectangle<int32_t>> available_list;
  // TODO.
  return available_list;
}

void GridManager::obtainOverflowIllegalGridList(std::vector<Grid*>& gird_list)
{
  // TODO.
}

void GridManager::clearAllOccupiedArea()
{
#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      _grid_2d_list[i][j].occupied_area = 0;
    }
  }
}

int64_t GridManager::obtainOverlapArea(Grid* grid, const Rectangle<int32_t>& rect)
{
  auto& grid_shape = grid->shape;

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

int64_t GridManager::obtainTotalOverflowArea()
{
  int64_t total_overflow_area = 0;

  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      total_overflow_area += _grid_2d_list[i][j].obtainGridOverflowArea();
    }
  }
}

float GridManager::obtainPeakGridDensity()
{
  float peak_density = __FLT_MIN__;

  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      float density = _grid_2d_list[i][j].obtainGridDensity();
      density > peak_density ? peak_density = density : density;
    }
  }

  return peak_density;
}

void GridManager::init()
{
  _grid_size_x = std::ceil(static_cast<float>(_shape.get_width()) / _grid_cnt_x);
  _grid_size_y = std::ceil(static_cast<float>(_shape.get_height()) / _grid_cnt_y);

  _grid_2d_list.resize(_grid_cnt_y);
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    _grid_2d_list[i].resize(_grid_cnt_x);
  }

#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      Grid cur_grid(i, j, _grid_size_x, _grid_size_y);
      int32_t y_coordi = _shape.get_ll_y() + i * _grid_size_y;
      int32_t x_coordi = _shape.get_ll_x() + j * _grid_size_x;
      cur_grid.shape = std::move(Rectangle<int32_t>(x_coordi, y_coordi, x_coordi + _grid_size_x, y_coordi + _grid_size_y));
      cur_grid.available_ratio = _available_ratio;

      _grid_2d_list[i][j] = std::move(cur_grid);
    }
  }
}

}  // namespace ipl