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
 * @Date: 2022-03-09 19:41:31
 * @LastEditTime: 2022-04-06 14:39:32
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/density/ElectricFieldGradient.cc
 * Contact : https://github.com/sjchanson
 */

#include "ElectricFieldGradient.hh"

#include "omp.h"
#include "usage/usage.hh"

namespace ipl {

void ElectricFieldGradient::initElectro2DList()
{
  size_t row_cnt = static_cast<size_t>(_grid_manager->obtainRowCntY());
  size_t grid_cnt = static_cast<size_t>(_grid_manager->obtainGridCntX());
  // _force_2d_list.resize(row_cnt);
  _force_2d_x_list.resize(row_cnt);
  _force_2d_y_list.resize(row_cnt);
  _phi_2d_list.resize(row_cnt);
  for (size_t i = 0; i < row_cnt; i++) {
    // _force_2d_list[i].resize(grid_cnt, std::make_pair(0.0f, 0.0f));
    _force_2d_x_list[i].resize(grid_cnt, 0.0f);
    _force_2d_y_list[i].resize(grid_cnt, 0.0f);
    _phi_2d_list[i].resize(grid_cnt, 0.0f);
  }
}

void ElectricFieldGradient::updateDensityForce(int32_t thread_num)
{
  // reset all variables.
  this->reset();

// copy density to utilize FFT
#pragma omp parallel for num_threads(thread_num)
  for (auto* row : _grid_manager->get_row_list()) {
    for (auto* grid : row->get_grid_list()) {
      // _fft->updateDensity(grid->get_grid_idx(), grid->get_row_idx(), grid->obtainGridDensity() / grid->get_available_ratio());
      _dct->updateDensity(grid->get_row_idx(), grid->get_grid_idx(), grid->obtainGridDensity() / grid->get_available_ratio());
    }
  }

  // do FFT
  // _fft->doFFT();
  ieda::Stats do_DCT_status;
  _dct->set_thread_nums(thread_num);
  _dct->doDCT(false);

  LOG_WARNING << "do_DCT runtime: " << do_DCT_status.elapsedRunTime() << " s";

  // update electro phi and electro force
  // update _sum_phi for nesterov loop
#pragma omp parallel for num_threads(thread_num)
  for (auto* row : _grid_manager->get_row_list()) {
    int32_t row_idx = row->get_row_idx();

    for (auto* grid : row->get_grid_list()) {
      int32_t grid_idx = grid->get_grid_idx();

      std::pair<float, float> e_force_pair = _dct->get_electro_force(grid->get_grid_idx(), grid->get_row_idx());
      _force_2d_x_list[row_idx][grid_idx] = e_force_pair.first;
      _force_2d_y_list[row_idx][grid_idx] = e_force_pair.second;

      // float electro_phi = _fft->get_electro_phi(grid->get_grid_idx(), grid->get_row_idx());
      float electro_phi = _dct->get_electro_phi(grid->get_row_idx(), grid->get_grid_idx());
      _phi_2d_list[row_idx][grid_idx] = electro_phi;

      _sum_phi += electro_phi * static_cast<float>(grid->get_occupied_area());
    }
  }
}

Point<float> ElectricFieldGradient::obtainDensityGradient(Rectangle<int32_t> shape, float scale)
{
  float gradient_x = 0;
  float gradient_y = 0;

  std::vector<Grid*> overlap_grid_list;
  _grid_manager->obtainOverlapGridList(overlap_grid_list, shape);
  for (auto* grid : overlap_grid_list) {
    float overlap_area = _grid_manager->obtainOverlapArea(grid, shape) * scale;

    gradient_x += overlap_area * _force_2d_x_list[grid->get_row_idx()][grid->get_grid_idx()];
    gradient_y += overlap_area * _force_2d_y_list[grid->get_row_idx()][grid->get_grid_idx()];
  }
  return Point<float>(gradient_x, gradient_y);
}

void ElectricFieldGradient::reset()
{
  _sum_phi = 0;
}

}  // namespace ipl