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
 * @Date: 2022-02-28 14:58:19
 * @LastEditTime: 2022-11-30 09:55:05
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/grid_manager/GridManager.cc
 * Contact : https://github.com/sjchanson
 */

#include "GridManager.hh"

#include "math.h"

namespace ipl {

void Grid::set_shape(Rectangle<int32_t> shape)
{
  _width = shape.get_width();
  _height = shape.get_height();
  _shape = std::move(shape);
}

int64_t Grid::obtainAvailableArea()
{
  int64_t width = static_cast<int64_t>(this->get_width());
  int64_t height = static_cast<int64_t>(this->get_height());
  return static_cast<int64_t>(width * height * _available_ratio) - this->get_occupied_area() - this->get_fixed_area();
}

void Grid::clearOccupiedArea()
{
  _occupied_area = 0;
}

void Grid::clearFixedArea()
{
  _fixed_area = 0;
}

void Grid::regainFreeArea(int64_t free_area)
{
  _occupied_area -= free_area;
  LOG_ERROR_IF(_occupied_area < 0) << "Grid : " << _row_idx << "," << _grid_idx << " _occupied_area is less than zero!";
}

float Grid::obtainGridDensity()
{
  int64_t width = static_cast<int64_t>(_shape.get_width());
  int64_t height = static_cast<int64_t>(_shape.get_height());

  return static_cast<float>(_occupied_area + _fixed_area) / (width * height);
}

int64_t Grid::obtainGridOverflowArea()
{
  int64_t width = static_cast<int64_t>(_shape.get_width());
  int64_t height = static_cast<int64_t>(_shape.get_height());
  return std::max(int64_t(0), static_cast<int64_t>(_occupied_area + _fixed_area - (_available_ratio * width * height)));
}

// std::vector<Grid*> GridRow::obtainOverlapGridList(int32_t lower_x, int32_t upper_x)
// {
//   LOG_ERROR_IF(lower_x > upper_x) << " Inquiry GridList Range is unlegal!";

//   std::vector<Grid*> overlap_list;
//   int32_t            grid_width = _grid_list[0]->get_shape().get_width();
//   int32_t            grid_cnt   = _grid_list.size();

//   std::pair<int, int> idx_range = obtainMinMaxIdx(_shape.get_ll_x(), grid_width, lower_x, upper_x);

//   int32_t lower_idx = idx_range.first;
//   int32_t upper_idx = idx_range.second;

//   LOG_ERROR_IF(lower_idx < 0 || upper_idx > grid_cnt) << " Inquiry GridList upper index is out of Row boundary!";

//   for (int i = lower_idx; i < upper_idx; i++) {
//     overlap_list.emplace_back(_grid_list[i]);
//   }

//   return overlap_list;
// }

void GridRow::obtainOverlapGridList(std::vector<Grid*>& overlap_list, int32_t lower_x, int32_t upper_x)
{
  LOG_ERROR_IF(lower_x > upper_x) << " Inquiry GridList Range is unlegal!";
  LOG_ERROR_IF(!overlap_list.empty()) << "Pass overlap_list is not Empty!";

  Utility utility;
  int32_t grid_width = _grid_list[0]->get_width();
  int32_t grid_cnt = _grid_list.size();

  std::pair<int, int> idx_range = utility.obtainMinMaxIdx(_shape.get_ll_x(), grid_width, lower_x, upper_x);

  int32_t lower_idx = idx_range.first;
  int32_t upper_idx = idx_range.second;

  if (lower_idx < 0 || upper_idx > grid_cnt) {
    LOG_ERROR << " Inquiry GridList upper index is out of Row boundary!";
  }

  for (int i = lower_idx; i < upper_idx; i++) {
    overlap_list.emplace_back(_grid_list[i]);
  }
}

std::vector<Grid*> GridRow::obtainAvailableGridList(float available_ratio)
{
  LOG_ERROR_IF(available_ratio < 0) << "Available ratio is less than zero!";

  std::vector<Grid*> available_grid_list;

  int64_t grid_width = static_cast<int64_t>(_grid_list[0]->get_width());
  int64_t grid_height = static_cast<int64_t>(_grid_list[0]->get_height());

  int64_t grid_area = grid_width * grid_height;
  int64_t required_area = static_cast<int64_t>(grid_area * available_ratio);

  for (auto* grid : _grid_list) {
    if (grid->obtainAvailableArea() >= required_area) {
      available_grid_list.push_back(grid);
    }
  }

  return available_grid_list;
}

std::vector<Grid*> GridRow::obtainAvailableGridList(int32_t grid_left, int32_t grid_right, float available_ratio)
{
  LOG_ERROR_IF(available_ratio < 0) << "Available ratio is less than zero!";

  std::vector<Grid*> available_grid_list;

  int64_t grid_width = static_cast<int64_t>(_grid_list[0]->get_width());
  int64_t grid_height = static_cast<int64_t>(_grid_list[0]->get_height());

  int64_t grid_area = grid_width * grid_height;
  int64_t required_area = static_cast<int64_t>(grid_area * available_ratio);

  for (int32_t i = grid_left; i < grid_right; i++) {
    if (_grid_list.at(i)->obtainAvailableArea() >= required_area) {
      available_grid_list.push_back(_grid_list.at(i));
    }
  }

  return available_grid_list;
}

std::vector<Rectangle<int32_t>> GridRow::obtainAvailableGridShapeList(float available_ratio)
{
  LOG_ERROR_IF(available_ratio < 0) << "Available ratio is less than zero!";

  std::vector<Rectangle<int32_t>> available_shape_list;
  std::vector<Grid*> available_grid_list = this->obtainAvailableGridList(available_ratio);

  for (size_t i = 0, j = i; i < available_grid_list.size();) {
    if (j + 1 >= available_grid_list.size()) {
      available_shape_list.push_back((available_grid_list.back())->get_shape());
      break;
    }

    while (available_grid_list[j + 1]->get_grid_idx() - available_grid_list[j]->get_grid_idx() == 1) {
      j++;
      if (j + 1 >= available_grid_list.size()) {
        break;
      }
    }

    available_shape_list.push_back(
        Rectangle<int32_t>(available_grid_list[i]->get_shape().get_lower_left(), available_grid_list[j]->get_shape().get_upper_right()));
    i = ++j;  // next.
  }

  return available_shape_list;
}

std::vector<Rectangle<int32_t>> GridRow::calContinuouslyShapeList(std::vector<Grid*> available_grid_list)
{
  std::vector<Rectangle<int32_t>> available_shape_list;

  for (size_t i = 0, j = i; i < available_grid_list.size();) {
    if (j + 1 >= available_grid_list.size()) {
      available_shape_list.push_back((available_grid_list.back())->get_shape());
      break;
    }

    while (available_grid_list[j + 1]->get_grid_idx() - available_grid_list[j]->get_grid_idx() == 1) {
      j++;
      if (j + 1 >= available_grid_list.size()) {
        break;
      }
    }

    available_shape_list.push_back(
        Rectangle<int32_t>(available_grid_list[i]->get_shape().get_lower_left(), available_grid_list[j]->get_shape().get_upper_right()));
    i = ++j;  // next.
  }

  return available_shape_list;
}

void GridRow::clearAllOccupiedArea()
{
  for (auto* grid : _grid_list) {
    grid->clearOccupiedArea();
  }
}

int32_t GridManager::obtainGridCntX()
{
  return _row_list[0]->get_grid_list().size();
}

int32_t GridManager::obtainRowCntY()
{
  return _row_list.size();
}

// std::vector<GridRow*> GridManager::obtainOverlapRowList(int32_t lower_y, int32_t upper_y)
// {
//   LOG_ERROR_IF(lower_y > upper_y) << " Inquiry GridList Range is unlegal!";

//   std::vector<GridRow*> overlap_row_list;
//   int32_t               row_height = _row_list[0]->get_shape().get_height();
//   int32_t               row_cnt    = _row_list.size();

//   std::pair<int, int> idx_range = obtainMinMaxIdx(_shape.get_ll_y(), row_height, lower_y, upper_y);
//   int32_t             lower_idx = idx_range.first;
//   int32_t             upper_idx = idx_range.second;

//   LOG_ERROR_IF(lower_idx < 0 || upper_idx > row_cnt) << " Inquiry GridList upper index is out of Row boundary!";

//   for (int i = lower_idx; i < upper_idx; i++) {
//     overlap_row_list.emplace_back(_row_list[i]);
//   }

//   return overlap_row_list;
// }

void GridManager::obtainOverlapRowList(std::vector<GridRow*>& overlap_row_list, int32_t lower_y, int32_t upper_y)
{
  LOG_ERROR_IF(lower_y > upper_y) << " Inquiry GridList Range is unlegal!";

  if (lower_y > upper_y) {
    LOG_WARNING << "DEBUG : " << lower_y << " , " << upper_y;
  }

  Utility utility;
  int32_t row_height = this->get_grid_size_y();
  int32_t row_cnt = _row_list.size();

  std::pair<int, int> idx_range = utility.obtainMinMaxIdx(_shape.get_ll_y(), row_height, lower_y, upper_y);
  int32_t lower_idx = idx_range.first;
  int32_t upper_idx = idx_range.second;

  if (lower_idx < 0 || upper_idx > row_cnt) {
    LOG_ERROR << " Inquiry GridList upper index is out of Row boundary!";
  }

  for (int i = lower_idx; i < upper_idx; i++) {
    overlap_row_list.emplace_back(_row_list[i]);
  }
}

// std::vector<Grid*> GridManager::obtainOverlapGridList(Rectangle<int32_t> rect)
// {
//   // for avg grid assignment.
//   auto    grid_shape  = _row_list[0]->get_grid_list()[0]->get_shape();
//   int32_t grid_width  = grid_shape.get_width();
//   int32_t grid_height = grid_shape.get_height();
//   int32_t rect_width  = rect.get_width();
//   int32_t rect_height = rect.get_height();
//   int32_t cnt_x       = (rect_width < grid_width) ? 2 : (rect_width / grid_width) + 1;
//   int32_t cnt_y       = (rect_height < grid_height) ? 2 : (rect_height / grid_height) + 1;

//   std::vector<Grid*> overlap_grid_list;
//   overlap_grid_list.reserve(cnt_x * cnt_y);

//   std::vector<GridRow*> grid_row_list = this->obtainOverlapRowList(rect.get_ll_y(), rect.get_ur_y());

//   for (auto* row : grid_row_list) {
//     std::vector<Grid*> grid_list = row->obtainOverlapGridList(rect.get_ll_x(), rect.get_ur_x());
//     overlap_grid_list.insert(overlap_grid_list.end(), grid_list.begin(), grid_list.end());
//   }

//   return overlap_grid_list;
// }

void GridManager::obtainOverlapGridList(std::vector<Grid*>& overlap_grid_list, Rectangle<int32_t>& rect)
{
  LOG_ERROR_IF(!overlap_grid_list.empty()) << "Pass overlap_grid_list is not Empty!";

  int32_t rect_width = rect.get_width();
  int32_t rect_height = rect.get_height();
  int32_t cnt_x = (rect_width < _grid_size_x) ? 2 : (rect_width / _grid_size_x) + 1;
  int32_t cnt_y = (rect_height < _grid_size_y) ? 2 : (rect_height / _grid_size_y) + 1;
  overlap_grid_list.reserve(cnt_x * cnt_y);

  std::vector<GridRow*> grid_row_list;
  grid_row_list.reserve(cnt_y);
  obtainOverlapRowList(grid_row_list, rect.get_ll_y(), rect.get_ur_y());

  for (auto* row : grid_row_list) {
    std::vector<Grid*> grid_list;
    grid_list.reserve(cnt_x);
    row->obtainOverlapGridList(grid_list, rect.get_ll_x(), rect.get_ur_x());
    overlap_grid_list.insert(overlap_grid_list.end(), grid_list.begin(), grid_list.end());
  }
}

std::vector<Rectangle<int32_t>> GridManager::obtainAvailableRectList(int32_t row_low, int32_t row_high, int32_t grid_left,
                                                                     int32_t grid_right, float available_ratio)
{
  std::vector<Rectangle<int32_t>> available_list;
  for (int i = row_low; i < row_high; i++) {
    auto* grid_row = _row_list.at(i);
    std::vector<Grid*> available_grid_list = grid_row->obtainAvailableGridList(grid_left, grid_right, available_ratio);
    std::vector<Rectangle<int32_t>> tmp_list = grid_row->calContinuouslyShapeList(available_grid_list);
    available_list.insert(available_list.end(), tmp_list.begin(), tmp_list.end());
  }

  return available_list;
}

std::vector<Grid*> GridManager::obtainOverflowIllegalGridList()
{
  std::vector<Grid*> overflow_grid_list;

  for (auto* row : _row_list) {
    for (auto* grid : row->get_grid_list()) {
      int64_t overflow_area = grid->obtainGridOverflowArea();
      if (overflow_area > 0) {
        overflow_grid_list.push_back(grid);
      }
    }
  }

  return overflow_grid_list;
}

void GridManager::clearAllOccupiedArea()
{
  for (auto* row : _row_list) {
    row->clearAllOccupiedArea();
  }
}

void GridManager::init(int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio)
{
  int32_t grid_size_y = std::ceil(static_cast<float>(_shape.get_height()) / grid_cnt_y);
  int32_t grid_size_x = std::ceil(static_cast<float>(_shape.get_width()) / grid_cnt_x);

  _grid_size_x = grid_size_x;
  _grid_size_y = grid_size_y;

  int32_t y_coord = _shape.get_ll_y();
  for (int32_t i = 0; i < grid_cnt_y; i++) {
    GridRow* grid_row = new GridRow(i);

    // set shape.
    int32_t size_y = (y_coord + grid_size_y > _shape.get_ur_y()) ? _shape.get_ur_y() - y_coord : grid_size_y;
    grid_row->set_shape(Rectangle<int32_t>(_shape.get_ll_x(), y_coord, _shape.get_ur_x(), y_coord + size_y));

    int32_t x_coord = _shape.get_ll_x();
    for (int32_t j = 0; j < grid_cnt_x; j++) {
      Grid* grid = new Grid(i, j);

      // set shape.
      int32_t size_x = (x_coord + grid_size_x > _shape.get_ur_x()) ? _shape.get_ur_x() - x_coord : grid_size_x;
      grid->set_shape(Rectangle<int32_t>(x_coord, y_coord, x_coord + size_x, y_coord + size_y));

      // set ratio.
      grid->set_available_ratio(available_ratio);

      grid_row->add_grid_list(grid);
      x_coord += grid_size_x;
    }

    this->add_row_list(grid_row);
    y_coord += grid_size_y;
  }
}

}  // namespace ipl