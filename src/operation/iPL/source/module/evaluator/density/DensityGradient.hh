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
 * @Date: 2022-03-09 16:47:44
 * @LastEditTime: 2022-11-23 12:12:58
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/density/DensityGradient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_DENSITY_GRADIENT_H
#define IPL_EVALUATOR_DENSITY_GRADIENT_H

#include "GridManager.hh"

namespace ipl {

class DensityGradient
{
 public:
  DensityGradient() = delete;
  explicit DensityGradient(GridManager* grid_manager);
  DensityGradient(const DensityGradient&) = delete;
  DensityGradient(DensityGradient&&) = delete;
  virtual ~DensityGradient() = default;

  DensityGradient& operator=(const DensityGradient&) = delete;
  DensityGradient& operator=(DensityGradient&&) = delete;

  virtual void updateDensityForce(int32_t thread_num, bool is_cal_phi) = 0;
  virtual Point<float> obtainDensityGradient(Rectangle<int32_t> shape, float scale, bool is_add_quad_penalty, float quad_lamda) = 0;

  // tmp for debug
  virtual std::vector<std::vector<float>>& get_force_2d_x_list() = 0;
  virtual std::vector<std::vector<float>>& get_force_2d_y_list() = 0;

  virtual float get_sum_phi() = 0;

 protected:
  GridManager* _grid_manager;
};
inline DensityGradient::DensityGradient(GridManager* grid_manager) : _grid_manager(grid_manager)
{
}

}  // namespace ipl

#endif