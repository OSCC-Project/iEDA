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
#include <fstream>

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
  if (row_low < 0) {
    LOG_WARNING << "Required Available Rectangle Row Range Less Than Zero!";
    row_low = 0;
  }
  if (row_high > (_grid_cnt_y - 1)) {
    LOG_WARNING << "Required Available Rectangle Row Range More Than (GridCntY-1)!";
    row_high = (_grid_cnt_y - 1);
  }
  if (grid_left < 0) {
    LOG_WARNING << "Required Available Rectangle Column Range Less Than Zero!";
    grid_left = 0;
  }
  if (grid_right > (_grid_cnt_x - 1)) {
    LOG_WARNING << "Required Available Rectangle Column Range More Than (GridCntX-1)!";
    grid_right = (_grid_cnt_x - 1);
  }
  int64_t target_available_area = static_cast<int64_t>(_grid_size_x) * static_cast<int64_t>(_grid_size_y) * available_ratio;

  std::vector<Rectangle<int32_t>> available_list;
  for (int32_t i = row_low; i <= row_high; i++) {
    int32_t lly = row_low * _grid_size_y;
    int32_t ury = lly + _grid_size_y;
    int32_t llx_idx = grid_left;
    int32_t urx_idx = llx_idx;
    for (int32_t j = grid_left; j <= grid_right; j++) {
      if (_grid_2d_list[i][j].obtainAvailableArea() < target_available_area) {
        if (llx_idx != urx_idx) {
          available_list.push_back(Rectangle<int32_t>(llx_idx * _grid_size_x, lly, urx_idx * _grid_size_x, ury));
        }
        llx_idx = j + 1;  // next idx
        urx_idx = llx_idx;
      } else {
        urx_idx = j + 1;
      }
    }
  }

  return available_list;
}

void GridManager::obtainOverflowIllegalGridList(std::vector<Grid*>& gird_list)
{
  for (auto& row_vec : _grid_2d_list) {
    for (auto& grid : row_vec) {
      int64_t overflow_area = grid.obtainGridOverflowArea();
      if (overflow_area > 0) {
        gird_list.push_back(&grid);
      }
    }
  }
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

void GridManager::clearAllOccupiedNodeNum()
{
#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      _grid_2d_list[i][j].num_node = 0;
    }
  }
}

void GridManager::clearRUDY()
{
#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      _grid_2d_list[i][j].h_cong = 0.0;
      _grid_2d_list[i][j].v_cong = 0.0;
    }
  }  
}

void GridManager::initRouteCap(int32_t h_cap, int32_t v_cap)
{
#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      _grid_2d_list[i][j].h_cap = h_cap;
      _grid_2d_list[i][j].v_cap = v_cap;
    }
  }  
}

void GridManager::evalRouteUtil()
{
  _h_util_max = 0.0f;
  _v_util_max = 0.0f;
  _h_util_sum = 0.0f;
  _v_util_sum = 0.0f;
#pragma omp parallel for num_threads(_thread_num)
  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      _grid_2d_list[i][j].h_util = _grid_2d_list[i][j].h_cong / _grid_2d_list[i][j].h_cap;
      _grid_2d_list[i][j].v_util = _grid_2d_list[i][j].v_cong / _grid_2d_list[i][j].v_cap;
            
      #pragma omp critical
      {
          if (_grid_2d_list[i][j].h_util > _h_util_max) {
              _h_util_max = _grid_2d_list[i][j].h_util;
          }
          
          if (_grid_2d_list[i][j].v_util > _v_util_max) {
              _v_util_max = _grid_2d_list[i][j].v_util;
          }
          _h_util_sum += _grid_2d_list[i][j].h_util;
          _v_util_sum += _grid_2d_list[i][j].v_util;
      }
    }
  }
}

void GridManager::blurRouteDemand()
{
  int height = _grid_cnt_y;
  int width = _grid_cnt_x;
  std::vector<std::vector<float>> paddedH(height + 2, std::vector<float>(width + 2, 0.0));
  std::vector<std::vector<float>> paddedV(height + 2, std::vector<float>(width + 2, 0.0));

#pragma omp parallel for num_threads(_thread_num)
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          paddedH[i + 1][j + 1] = _grid_2d_list[i][j].h_cong;
          paddedV[i + 1][j + 1] = _grid_2d_list[i][j].v_cong;
      }
  }

  int kernelSize = 3;
  float h_sigma = 5;
  float v_sigma = 5;
  fastGaussianBlur(paddedH, h_sigma, kernelSize);
  fastGaussianBlur(paddedV, v_sigma, kernelSize);

#pragma omp parallel for num_threads(_thread_num)
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          _grid_2d_list[i][j].h_cong = paddedH[i+1][j+1];
          _grid_2d_list[i][j].v_cong = paddedV[i+1][j+1];
      }
  }

}

void GridManager::fastGaussianBlur(std::vector<std::vector<float>>& image, float sigma, int kernelSize)
{
  int height = image.size();
  int width = image[0].size();
  int paddingSize = kernelSize / 2;

  std::vector<std::vector<float>> blurredImage(height, std::vector<float>(width, 0.0));

#pragma omp parallel for num_threads(_thread_num)
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          double sum = 0.0;
          double weightSum = 0.0;
          for (int k = -paddingSize; k <= paddingSize; k++) {
              int y = i + k;
              if (y >= 0 && y < height) {
                  double weight = exp(-(k * k) / (2 * sigma * sigma));
                  sum += image[y][j] * weight;
                  weightSum += weight;
              }
          }
          blurredImage[i][j] = sum / weightSum;
      }
  }
  std::vector<std::vector<float>> tempImage(height, std::vector<float>(width, 0.0));

#pragma omp parallel for num_threads(_thread_num)
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          double sum = 0.0;
          double weightSum = 0.0;
          for (int k = -paddingSize; k <= paddingSize; k++) {
              int x = j + k;
              if (x >= 0 && x < width) {
                  double weight = exp(-(k * k) / (2 * sigma * sigma));
                  sum += blurredImage[i][x] * weight;
                  weightSum += weight;
              }
          }
          tempImage[i][j] = sum / weightSum;
      }
  }

#pragma omp parallel for num_threads(_thread_num)
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          image[i][j] = tempImage[i][j];
      }
  }
}


void GridManager::plotRouteCap()
{
  std::ofstream plot_h("h_cap.csv");
  std::ofstream plot_v("v_cap.csv");
  std::stringstream feed_h;
  std::stringstream feed_v;
  int32_t x_cnt = _grid_cnt_x;
  int32_t y_cnt = _grid_cnt_y;

  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed_h << "col_" << i;
      feed_v << "col_" << i;
    } else {
      feed_h << "col_" << i << ",";
      feed_v << "col_" << i << ",";
    }
  }
  feed_h << std::endl;
  feed_v << std::endl;

  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      int h_cap = _grid_2d_list[i][j].h_cap;
      int v_cap = _grid_2d_list[i][j].v_cap;
      if (j == x_cnt - 1) {
        feed_h << h_cap;
        feed_v << v_cap;
      } else {
        feed_h << h_cap << ",";
        feed_v << v_cap << ",";
      }
    }
    feed_h << std::endl;
    feed_v << std::endl;
  }
  
  plot_h << feed_h.str();
  plot_v << feed_v.str();
  feed_h.clear();
  plot_h.close();
  feed_v.clear();
  plot_v.close();
}

void GridManager::plotRouteDem()
{
  std::ofstream plot_h("h_dem.csv");
  std::ofstream plot_v("v_dem.csv");
  std::stringstream feed_h;
  std::stringstream feed_v;
  int32_t x_cnt = _grid_cnt_x;
  int32_t y_cnt = _grid_cnt_y;

  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed_h << "col_" << i;
      feed_v << "col_" << i;
    } else {
      feed_h << "col_" << i << ",";
      feed_v << "col_" << i << ",";
    }
  }
  feed_h << std::endl;
  feed_v << std::endl;

  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      int h_cong = _grid_2d_list[i][j].h_cong;
      int v_cong = _grid_2d_list[i][j].v_cong;
      if (j == x_cnt - 1) {
        feed_h << h_cong;
        feed_v << v_cong;
      } else {
        feed_h << h_cong << ",";
        feed_v << v_cong << ",";
      }
    }
    feed_h << std::endl;
    feed_v << std::endl;
  }
  
  plot_h << feed_h.str();
  plot_v << feed_v.str();
  feed_h.clear();
  plot_h.close();
  feed_v.clear();
  plot_v.close();
}

void GridManager::plotRouteUtil(int32_t iter_num)
{
  std::ofstream plot_h("h_util_"+std::to_string(iter_num)+".csv");
  std::ofstream plot_v("v_util_"+std::to_string(iter_num)+".csv");
  std::stringstream feed_h;
  std::stringstream feed_v;
  int32_t x_cnt = _grid_cnt_x;
  int32_t y_cnt = _grid_cnt_y;

  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed_h << "col_" << i;
      feed_v << "col_" << i;
    } else {
      feed_h << "col_" << i << ",";
      feed_v << "col_" << i << ",";
    }
  }
  feed_h << std::endl;
  feed_v << std::endl;

  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      float h_cong = _grid_2d_list[i][j].h_util;
      float v_cong = _grid_2d_list[i][j].v_util;
      if (j == x_cnt - 1) {
        feed_h << h_cong;
        feed_v << v_cong;
      } else {
        feed_h << h_cong << ",";
        feed_v << v_cong << ",";
      }
    }
    feed_h << std::endl;
    feed_v << std::endl;
  }
  
  plot_h << feed_h.str();
  plot_v << feed_v.str();
  feed_h.clear();
  plot_h.close();
  feed_v.clear();
  plot_v.close();
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

Rectangle<int32_t> GridManager::obtainOverlapRect(Grid* grid, const Rectangle<int32_t>& rect)
{
  auto& grid_shape = grid->shape;

  int32_t overlap_rect_lx = std::max(grid_shape.get_ll_x(), rect.get_ll_x());
  int32_t overlap_rect_ly = std::max(grid_shape.get_ll_y(), rect.get_ll_y());
  int32_t overlap_rect_ux = std::min(grid_shape.get_ur_x(), rect.get_ur_x());
  int32_t overlap_rect_uy = std::min(grid_shape.get_ur_y(), rect.get_ur_y());

  if (overlap_rect_lx >= overlap_rect_ux || overlap_rect_ly >= overlap_rect_uy) {
    int32_t fake_rect_lx = (overlap_rect_lx + overlap_rect_ux) / 2;
    int32_t fake_rect_ly = (overlap_rect_ly + overlap_rect_uy) / 2;
    Rectangle rect(fake_rect_lx, fake_rect_ly, fake_rect_lx, fake_rect_ly);
    return rect;
  } else {
    Rectangle rect(overlap_rect_lx,overlap_rect_ly,overlap_rect_ux,overlap_rect_uy);
    return rect;
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
  return total_overflow_area;
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

float GridManager::obtainAvgGridDensity(){
  float sum_density = 0.0f;

  for (int32_t i = 0; i < _grid_cnt_y; i++) {
    for (int32_t j = 0; j < _grid_cnt_x; j++) {
      float density = _grid_2d_list[i][j].obtainGridDensity();
      sum_density += density;
    }
  }

  int64_t grid_cnt = _grid_cnt_x * _grid_cnt_y;
  return (sum_density / grid_cnt);
}

void GridManager::init()
{
  if (_grid_size_x == -1 && _grid_size_y == -1){
    _grid_size_x = std::ceil(static_cast<float>(_shape.get_width()) / _grid_cnt_x);
    _grid_size_y = std::ceil(static_cast<float>(_shape.get_height()) / _grid_cnt_y);
  }

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