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

#include <cassert>
#include <iostream>
#include <vector>

#include "GridManager.hh"
#include "NesInstance.hh"
#include "Parameter.hh"
#include "TopologyManager.hh"

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
  void set_route_cap_h(int num) { _route_cap_h = num; }
  void set_route_cap_v(int num) { _route_cap_v = num; }
  void set_partial_route_cap_h(int num) { _partial_route_cap_h = num; }
  void set_partial_route_cap_v(int num) { _partial_route_cap_v = num; }

  void initNesInstanceTypeList(std::vector<NesInstance*>& nInst_list);

  void updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num);
  void updataOverflowArea(std::vector<NesInstance*>& nInst_list, int32_t thread_num);

  void evalRouteDem(const std::vector<NetWork*>& network_list, int32_t thread_num);
  void evalRouteCap(int32_t thread_num);
  void evalRouteUtil();
  void plotRouteCap();
  void plotRouteUtil(int32_t iter_num);
  void plotRouteDem();
  void fastGaussianBlur();
  void plotOverflowUtil(float sum_overflow, int32_t iter_num);

  int64_t obtainOverflowAreaWithoutFiller();
  int64_t obtainOverflowArea();

  std::vector<Grid*> obtainOccupiedBinList(NesInstance* nInst);  // TODO.
  std::vector<NesInstance*> obtainOccupiednInstList(Grid* bin);  // TODO.
  GridManager* get_grid_manager() const { return _grid_manager; }

 private:
  GridManager* _grid_manager;
  int32_t _thread_nums;

  std::vector<NesInstance*> _macro_inst_list;
  std::vector<NesInstance*> _stdcell_list;
  std::vector<NesInstance*> _filler_list;
  std::vector<NesInstance*> _route_macro_inst_list;

  int64_t _overflow_area_wfiller;
  int64_t _overflow_area_wofiller;

  int32_t _bin_cnt_x;
  int32_t _bin_cnt_y;
  int32_t _bin_size_x;
  int32_t _bin_size_y;

  int _route_cap_h;
  int _route_cap_v;
  int _partial_route_cap_h;
  int _partial_route_cap_v;

  std::vector<std::vector<NesInstance*>> _bin_inst_list;
  std::vector<AreaInfo> _bin_area_list;

  void resetBinToArea();

  double calcLness(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax);
  int64_t calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymin);
  int64_t calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymin);
  int64_t calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymax);
  int64_t calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymax);

  void addBinnInstConnection(Grid* bin, NesInstance* nInst);
  void addBinMacroAreaInfo(Grid* bin, int64_t macro_area);
  void addBinStdcellAreaInfo(Grid* bin, int64_t stdcell_area);
  void addBinFillerAreaInfo(Grid* bin, int64_t filler_area);
};
inline BinGrid::BinGrid(GridManager* grid_manager)
    : _grid_manager(grid_manager), _thread_nums(1), _overflow_area_wfiller(INT64_MIN), _overflow_area_wofiller(INT64_MIN)
{
  _bin_cnt_x = _grid_manager->get_grid_cnt_x();
  _bin_cnt_y = _grid_manager->get_grid_cnt_y();
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
      if (nInst->isMacro()) {
        _route_macro_inst_list.push_back(nInst);
      }
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

inline void BinGrid::evalRouteDem(const std::vector<NetWork*>& network_list, int32_t thread_num)
{
  _grid_manager->clearRUDY();
  int wire_space_h = _bin_size_y / (_route_cap_h / _bin_cnt_y);
  int wire_space_v = _bin_size_x / (_route_cap_v / _bin_cnt_x);
  float dm_h = 4;
  float dm_v = 4;

#pragma omp parallel for num_threads(thread_num)
  for (auto* network : network_list) {
    if (network->isIgnoreNetwork()) {
      continue;
    }
    // FIXME: Temporarily fix this bug.
    auto net_shape = std::move(network->obtainNetWorkShape());
    auto old_shape = net_shape;
    if (old_shape.get_ll_x() > old_shape.get_ur_x() || old_shape.get_ll_y() > old_shape.get_ur_y()) {
      continue;
    }
    if (old_shape.get_ll_x() > _grid_manager->get_shape().get_ur_x() || old_shape.get_ll_y() > _grid_manager->get_shape().get_ur_y()
        || old_shape.get_ur_x() < _grid_manager->get_shape().get_ll_x() || old_shape.get_ur_y() < _grid_manager->get_shape().get_ll_y()) {
      continue;
    }
    double new_ur_x = std::min(net_shape.get_ur_x(), _grid_manager->get_shape().get_ur_x());
    double new_ur_y = std::min(net_shape.get_ur_y(), _grid_manager->get_shape().get_ur_y());
    net_shape.set_upper_right(new_ur_x, new_ur_y);

    double new_ll_x = std::max(net_shape.get_ll_x(), _grid_manager->get_shape().get_ll_x());
    double new_ll_y = std::max(net_shape.get_ll_y(), _grid_manager->get_shape().get_ll_y());
    net_shape.set_lower_left(new_ll_x, new_ll_y);

    assert(net_shape.get_ll_x() <= net_shape.get_ur_x());
    assert(net_shape.get_ll_y() <= net_shape.get_ur_y());

    int32_t pin_num = network->get_node_list().size();
    int64_t net_width = net_shape.get_width();
    int64_t net_height = net_shape.get_height();

    int32_t aspect_ratio = 1;
    if (net_width >= net_height && net_height != 0) {
      aspect_ratio = std::round(net_width / static_cast<double>(net_height));
    } else if (net_width < net_height && net_width != 0) {
      aspect_ratio = std::round(net_height / static_cast<double>(net_width));
    }

    double l_ness = 0.0;
    if (pin_num <= 3) {
      l_ness = 1.0;
    } else if (pin_num <= 15) {
      std::vector<std::pair<int32_t, int32_t>> point_set;
      point_set.reserve(pin_num);
      for (int i = 0; i < pin_num; ++i) {
        const int32_t pin_x = network->get_node_list()[i]->get_location().get_x();
        const int32_t pin_y = network->get_node_list()[i]->get_location().get_y();
        point_set.emplace_back(std::make_pair(pin_x, pin_y));
      }
      l_ness = calcLness(point_set, net_shape.get_ll_x(), net_shape.get_ur_x(), net_shape.get_ll_y(), net_shape.get_ur_y());
    } else {
      l_ness = 0.5;
    }
    l_ness = netWiringDistributionMapWeight(pin_num, aspect_ratio, l_ness);

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, net_shape);
    for (auto* grid : overlap_grid_list) {
      auto& grid_h_cong = grid->h_cong;
      auto& grid_v_cong = grid->v_cong;

      int64_t overlap_area;
      int rect_lx = std::max(grid->shape.get_ll_x(), net_shape.get_ll_x());
      int rect_ly = std::max(grid->shape.get_ll_y(), net_shape.get_ll_y());
      int rect_ux = std::min(grid->shape.get_ur_x(), net_shape.get_ur_x());
      int rect_uy = std::min(grid->shape.get_ur_y(), net_shape.get_ur_y());
      if (rect_lx >= rect_ux || rect_ly >= rect_uy) {
        overlap_area = 0;
      } else {
        overlap_area = (rect_ux - rect_lx) * (rect_uy - rect_ly);
      }

      float tmp_h_cong = 0.0;
      float tmp_v_cong = 0.0;
      if (net_shape.get_height() != 0) {
        tmp_h_cong = l_ness * overlap_area * wire_space_h * dm_h / static_cast<float>(net_shape.get_height());
      }
      if (net_shape.get_width() != 0) {
        tmp_v_cong = l_ness * overlap_area * wire_space_v * dm_v / static_cast<float>(net_shape.get_width());
      }

#pragma omp atomic
      grid_h_cong += tmp_h_cong;
#pragma omp atomic
      grid_v_cong += tmp_v_cong;
    }
  }
}

inline void BinGrid::fastGaussianBlur()
{
  _grid_manager->blurRouteDemand();
}

inline double BinGrid::calcLness(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t xmax, int32_t ymin,
                                 int32_t ymax)
{
  int64_t bbox = static_cast<int64_t>(xmax - xmin) * static_cast<int64_t>(ymax - ymin);
  int64_t r1 = calcLowerLeftRP(point_set, xmin, ymin);
  int64_t r2 = calcLowerRightRP(point_set, xmax, ymin);
  int64_t r3 = calcUpperLeftRP(point_set, xmin, ymax);
  int64_t r4 = calcUpperRightRP(point_set, xmax, ymax);
  int64_t r = std::max({r1, r2, r3, r4});
  double l_ness;
  if (bbox != 0) {
    l_ness = static_cast<double>(r) / static_cast<double>(bbox);
  } else {
    l_ness = 1.0;
  }
  return l_ness;
}

inline int64_t BinGrid::calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymin)
{
  std::sort(point_set.begin(), point_set.end());  // Sort point_set with x-coordinates in ascending order
  int64_t R = 0, y0 = point_set[0].second;
  for (size_t i = 1; i < point_set.size(); i++) {
    int32_t xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (xi - xmin) * (y0 - ymin));
      y0 = point_set[i].second;
    }
  }
  return R;
}

inline int64_t BinGrid::calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymin)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int64_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (xmax - xi) * (y0 - ymin));
      y0 = point_set[i].second;
    }
  }
  return R;
}

inline int64_t BinGrid::calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymax)
{
  std::sort(point_set.begin(), point_set.end(), [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
    return a.second > b.second;
  });  // Sort point_set with y-coordinates in descending order
  int64_t R = 0, x0 = point_set[0].first, yi;
  for (size_t i = 1; i < point_set.size(); i++) {
    yi = point_set[i].second;
    if (point_set[i].first <= x0) {
      R = std::max(R, (ymax - yi) * (x0 - xmin));
      x0 = point_set[i].first;
    }
  }
  return R;
}

inline int64_t BinGrid::calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymax)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int64_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second >= y0) {
      R = std::max(R, (ymax - y0) * (xmax - xi));
      y0 = point_set[i].second;
    }
  }
  return R;
}

inline void BinGrid::evalRouteCap(int32_t thread_num)
{
  int32_t bin_capa_h = _bin_size_x * _bin_size_y;
  int32_t bin_capa_v = bin_capa_h;
  _grid_manager->initRouteCap(bin_capa_h, bin_capa_v);

  float util_h = _partial_route_cap_h / (float) _route_cap_h;
  float util_v = _partial_route_cap_v / (float) _route_cap_v;

#pragma omp parallel for num_threads(thread_num)
  for (size_t i = 0; i < _route_macro_inst_list.size(); ++i) {
    auto macro_shape = _route_macro_inst_list[i]->get_origin_shape();

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, macro_shape);
    for (auto* grid : overlap_grid_list) {
      auto& grid_h_cap = grid->h_cap;
      auto& grid_v_cap = grid->v_cap;

      int64_t overlap_area;
      int rect_lx = std::max(grid->shape.get_ll_x(), macro_shape.get_ll_x());
      int rect_ly = std::max(grid->shape.get_ll_y(), macro_shape.get_ll_y());
      int rect_ux = std::min(grid->shape.get_ur_x(), macro_shape.get_ur_x());
      int rect_uy = std::min(grid->shape.get_ur_y(), macro_shape.get_ur_y());
      if (rect_lx >= rect_ux || rect_ly >= rect_uy) {
        overlap_area = 0;
      } else {
        overlap_area = (rect_ux - rect_lx) * (rect_uy - rect_ly);
      }
#pragma omp atomic
      grid_h_cap -= overlap_area * util_h * 1.2;
#pragma omp atomic
      grid_v_cap -= overlap_area * util_v * 1.2;
    }
  }
}

inline void BinGrid::evalRouteUtil()
{
  _grid_manager->evalRouteUtil();
}

inline void BinGrid::plotOverflowUtil(float sum_overflow, int32_t iter_num)
{
  std::ofstream plot_v("overflow_util.csv", std::ios::app);
  std::stringstream feed_v;
  feed_v << iter_num << "," << sum_overflow << "," << _grid_manager->get_h_util_max() << "," << _grid_manager->get_v_util_max();
  plot_v << feed_v.str() << std::endl;

  // std::ofstream plot_h("overflow_util_sum.csv", std::ios::app);
  // std::stringstream feed_h;
  // feed_h << iter_num << "," << sum_overflow << "," << _grid_manager->get_h_util_sum() << "," << _grid_manager->get_v_util_sum();
  // plot_h << feed_h.str() << std::endl;
}

inline void BinGrid::plotRouteCap()
{
  _grid_manager->plotRouteCap();
}

inline void BinGrid::plotRouteDem()
{
  _grid_manager->plotRouteDem();
}

inline void BinGrid::plotRouteUtil(int32_t iter_num)
{
  _grid_manager->plotRouteUtil(iter_num);
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
