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
  size_t row_cnt = static_cast<size_t>(_grid_manager->get_grid_cnt_y());
  size_t grid_cnt = static_cast<size_t>(_grid_manager->get_grid_cnt_x());
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

void ElectricFieldGradient::updateDensityForce(int32_t thread_num, bool is_cal_phi)
{
  // reset all variables.
  this->reset();

  // const auto& row_list = _grid_manager->get_row_list();

  // float** dct_density_map = _dct->get_density_2d_ptr();
  // float** dct_electro_x_map = _dct->get_electro_x_2d_ptr();
  // float** dct_electro_y_map = _dct->get_electro_y_2d_ptr();

  float** fft_density_map = _fft->get_density_2d_ptr();
  float** fft_electro_x_map = _fft->get_electro_x_2d_ptr();
  float** fft_electro_y_map = _fft->get_electro_y_2d_ptr();

  // copy density to utilize DCT
  int32_t grid_cnt_x = _grid_manager->get_grid_cnt_x();
  int32_t grid_cnt_y = _grid_manager->get_grid_cnt_y();
  float available_ratio = _grid_manager->get_available_ratio();
  auto& grid_2d_list = _grid_manager->get_grid_2d_list();

#pragma omp parallel for num_threads(thread_num)
  for (int32_t i = 0; i < grid_cnt_y; i++) {
    for (int32_t j = 0; j < grid_cnt_x; j++) {
      // dct_density_map[j][i] = grid_2d_list[i][j].obtainGridDensity() / available_ratio;
      fft_density_map[j][i] = grid_2d_list[i][j].obtainGridDensity() / available_ratio;
    }
  }

  // _dct->set_thread_nums(thread_num);
  // _dct->doDCT(is_cal_phi);
  _fft->set_thread_nums(thread_num);
  _fft->doFFT(is_cal_phi);

#pragma omp parallel for num_threads(thread_num)
  for (int32_t i = 0; i < grid_cnt_y; i++) {
    for (int32_t j = 0; j < grid_cnt_x; j++) {
      // _force_2d_x_list[i][j] = dct_electro_y_map[j][i];
      // _force_2d_y_list[i][j] = dct_electro_x_map[j][i];

      _force_2d_x_list[i][j] = fft_electro_x_map[j][i];
      _force_2d_y_list[i][j] = fft_electro_y_map[j][i];
    }
  }

  if (is_cal_phi) {
    // float** dct_phi_map = _dct->get_phi_2d_ptr();
    float** fft_phi_map = _fft->get_phi_2d_ptr();

    for (int32_t i = 0; i < grid_cnt_y; i++) {
      for (int32_t j = 0; j < grid_cnt_x; j++) {
        // float electro_phi = dct_phi_map[j][i];
        float electro_phi = fft_phi_map[j][i];
        _phi_2d_list[i][j] = electro_phi;
        _sum_phi += electro_phi * grid_2d_list[i][j].occupied_area;
      }
    }
  }
}

Point<float> ElectricFieldGradient::obtainDensityGradient(Rectangle<int32_t> shape, float scale, bool is_add_quad_penalty, float quad_lamda)
{
  float gradient_x = 0;
  float gradient_y = 0;

  std::vector<Grid*> overlap_grid_list;
  _grid_manager->obtainOverlapGridList(overlap_grid_list, shape);

  for (auto* grid : overlap_grid_list) {
    float overlap_area = _grid_manager->obtainOverlapArea(grid, shape) * scale;

    float grid_grad_x = overlap_area * _force_2d_x_list[grid->row_idx][grid->grid_idx];
    float grid_grad_y = overlap_area * _force_2d_y_list[grid->row_idx][grid->grid_idx];

    gradient_x += grid_grad_x;
    gradient_y += grid_grad_y;
  }
  return Point<float>(gradient_x, gradient_y);
}

void ElectricFieldGradient::reset()
{
  _sum_phi = 0;
}

}  // namespace ipl