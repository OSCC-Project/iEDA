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
  Grid() = default;
  Grid(int32_t r_id, int32_t g_id, int32_t w, int32_t h)
      : row_idx(r_id), grid_idx(g_id), width(w), height(h), available_ratio(1.0), occupied_area(0), fixed_area(0)
  {
    grid_area = static_cast<int64_t>(w) * static_cast<int64_t>(h);
  }
  Grid(const Grid&) = delete;
  Grid(Grid&& other) noexcept
  {
    row_idx = other.row_idx;
    grid_idx = other.grid_idx;
    width = other.width;
    height = other.height;
    grid_area = other.grid_area;
    shape = std::move(other.shape);
    available_ratio = other.available_ratio;
    occupied_area = other.occupied_area;
    fixed_area = other.fixed_area;
  }

  ~Grid() = default;

  Grid& operator=(const Grid&) = delete;
  Grid& operator=(Grid&& other)
  {
    if (this != &other) {
      row_idx = other.row_idx;
      grid_idx = other.grid_idx;
      width = other.width;
      height = other.height;
      grid_area = other.grid_area;
      shape = std::move(other.shape);
      available_ratio = other.available_ratio;
      occupied_area = other.occupied_area;
      fixed_area = other.fixed_area;
    }
    return (*this);
  }

  // function.
  int64_t obtainAvailableArea();
  int64_t obtainGridOverflowArea();
  float obtainGridDensity();
  std::vector<Grid*> &allNeighbors() {return neighbors;}
  void addNeighbor(Grid* grid) {neighbors.push_back(grid);}
  int getNumNodes() const { return num_node; }

  int32_t row_idx;
  int32_t grid_idx;

  int32_t width;
  int32_t height;
  int32_t grid_area;
  Rectangle<int32_t> shape;

  float available_ratio;
  int64_t occupied_area;
  int64_t fixed_area;
  int64_t placeable_area = 0;

  float h_cong;
  float v_cong;
  int32_t h_cap;
  int32_t v_cap;
  float h_util;
  float v_util;
  int num_node;

  std::vector<Grid*> neighbors;
};

class GridManager
{
 public:
  GridManager() = delete;
  GridManager(Rectangle<int32_t> region, int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio, int32_t thread_num);
  GridManager(Rectangle<int32_t> region, int32_t grid_cnt_x, int32_t grid_cnt_y, int32_t grid_size_x, int32_t grid_size_y, 
              float available_ratio, int32_t thread_num);
  GridManager(const GridManager&) = delete;
  GridManager(GridManager&&) = delete;
  ~GridManager();

  GridManager& operator=(const GridManager&) = delete;
  GridManager& operator=(GridManager&&) = delete;

  // getter.
  Rectangle<int32_t> get_shape() const { return _shape; }
  int32_t get_grid_cnt_x() const { return _grid_cnt_x; }
  int32_t get_grid_cnt_y() const { return _grid_cnt_y; }
  float get_available_ratio() const { return _available_ratio; }
  int32_t get_grid_size_x() const { return _grid_size_x; }
  int32_t get_grid_size_y() const { return _grid_size_y; }
  std::vector<std::vector<Grid>>& get_grid_2d_list() { return _grid_2d_list; }
  Utility get_utility() const {return _utility;}
  float get_h_util_max() const {return _h_util_max;}
  float get_v_util_max() const {return _v_util_max;}
  float get_h_util_sum() const {return _h_util_sum;}
  float get_v_util_sum() const {return _v_util_sum;}

  // function.
  void obtainOverlapGridList(std::vector<Grid*>& grid_list, Rectangle<int32_t>& rect);
  std::vector<Rectangle<int32_t>> obtainAvailableRectList(int32_t row_low, int32_t row_high, int32_t grid_left, int32_t grid_right,
                                                          float available_ratio);

  void obtainOverflowIllegalGridList(std::vector<Grid*>& grid_list);
  void clearAllOccupiedArea();
  void clearAllOccupiedNodeNum();

  void clearRUDY();
  void initRouteCap(int32_t h_cap, int32_t v_cap);
  void evalRouteUtil();
  void plotRouteCap();
  void plotRouteDem();
  void plotRouteUtil(int32_t iter_num);
  void fastGaussianBlur(std::vector<std::vector<float>>& image, float sigma, int kernelSize);
  void blurRouteDemand();

  int64_t obtainOverlapArea(Grid* grid, const Rectangle<int32_t>& rect);
  Rectangle<int32_t> obtainOverlapRect(Grid* grid, const Rectangle<int32_t>& rect);
  int64_t obtainTotalOverflowArea();
  float obtainPeakGridDensity();
  float obtainAvgGridDensity();

 private:
  Utility _utility;
  int32_t _thread_num;
  Rectangle<int32_t> _shape;
  int32_t _grid_cnt_x = -1;
  int32_t _grid_cnt_y = -1;
  int32_t _grid_size_x = -1;
  int32_t _grid_size_y = -1;
  float _available_ratio = 0.f;
  float _h_util_max = 0.f;
  float _v_util_max = 0.f;
  float _h_util_sum = 0.f;
  float _v_util_sum = 0.f;

  std::vector<std::vector<Grid>> _grid_2d_list;

  void init();
};

inline GridManager::GridManager(Rectangle<int32_t> region, int32_t grid_cnt_x, int32_t grid_cnt_y, float available_ratio,
                                int32_t thread_num)
    : _utility(Utility()),
      _thread_num(thread_num),
      _shape(std::move(region)),
      _grid_cnt_x(grid_cnt_x),
      _grid_cnt_y(grid_cnt_y),
      _available_ratio(available_ratio)
{
  init();
}

inline GridManager::GridManager(Rectangle<int32_t> region, int32_t grid_cnt_x, int32_t grid_cnt_y, int32_t grid_size_x, int32_t grid_size_y, 
              float available_ratio, int32_t thread_num)
    : _utility(Utility()),
      _thread_num(thread_num),
      _shape(std::move(region)),
      _grid_cnt_x(grid_cnt_x),
      _grid_cnt_y(grid_cnt_y),
      _grid_size_x(grid_size_x),
      _grid_size_y(grid_size_y),
      _available_ratio(available_ratio)
{
  init();
}

inline GridManager::~GridManager()
{
}

}  // namespace ipl

#endif