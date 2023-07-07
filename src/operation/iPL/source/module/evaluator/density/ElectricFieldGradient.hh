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
 * @Date: 2022-03-09 19:40:55
 * @LastEditTime: 2022-11-23 12:13:07
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/density/ElectricFieldGradient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_ELECTRIC_FIELD_GRADIENT_H
#define IPL_EVALUATOR_ELECTRIC_FIELD_GRADIENT_H

#include <map>
#include <unordered_map>
#include <vector>

#include "DensityGradient.hh"
#include "dct_process/FFT.hh"
// #include "dct_process/DCT.hh"

namespace ipl {

class ElectricFieldGradient : public DensityGradient
{
 public:
  ElectricFieldGradient() = delete;
  explicit ElectricFieldGradient(GridManager* grid_manager);
  ElectricFieldGradient(const ElectricFieldGradient&) = delete;
  ElectricFieldGradient(ElectricFieldGradient&&) = delete;
  ~ElectricFieldGradient() override = default;

  ElectricFieldGradient& operator=(const ElectricFieldGradient&) = delete;
  ElectricFieldGradient& operator=(ElectricFieldGradient&&) = delete;

  std::vector<std::vector<float>>& get_force_2d_x_list() override { return _force_2d_x_list; }
  std::vector<std::vector<float>>& get_force_2d_y_list() override { return _force_2d_y_list; }

  float get_sum_phi() override { return _sum_phi; }

  void updateDensityForce(int32_t thread_num, bool is_cal_phi) override;
  Point<float> obtainDensityGradient(Rectangle<int32_t> shape, float scale, bool is_add_quad_penalty, float quad_lamda) override;

  void reset();

 private:
  float _sum_phi;
  FFT* _fft;
  // DCT* _dct;

  std::vector<std::vector<float>> _force_2d_x_list;
  std::vector<std::vector<float>> _force_2d_y_list;
  // std::vector<std::vector<std::pair<float, float>>> _force_2d_list;
  std::vector<std::vector<float>> _phi_2d_list;
  void initElectro2DList();
};
inline ElectricFieldGradient::ElectricFieldGradient(GridManager* grid_manager) : DensityGradient(grid_manager), _sum_phi(0.0F)
{
  int32_t grid_size_x = grid_manager->get_grid_size_x();
  int32_t grid_size_y = grid_manager->get_grid_size_y();

  _fft = new FFT(_grid_manager->get_grid_cnt_x(), _grid_manager->get_grid_cnt_y(), grid_size_x, grid_size_y);
  // _dct = new DCT(_grid_manager->get_grid_cnt_x(), _grid_manager->get_grid_cnt_y(), grid_size_x, grid_size_y);

  initElectro2DList();
}

}  // namespace ipl

#endif