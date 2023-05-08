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

namespace ipl {

void ElectricFieldGradient::initElectroMap()
{
  for (auto* row : _grid_manager->get_row_list()) {
    for (auto* grid : row->get_grid_list()) {
      _electro_map.emplace(grid, ElectroInfo());
    }
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
      _fft->updateDensity(grid->get_grid_idx(), grid->get_row_idx(), grid->obtainGridDensity() / grid->get_available_ratio());
    }
  }

  // do FFT
  _fft->doFFT();

// update electro phi and electro force
// update _sum_phi for nesterov loop
#pragma omp parallel for num_threads(thread_num)
  for (auto* row : _grid_manager->get_row_list()) {
    for (auto* grid : row->get_grid_list()) {
      ElectroInfo electro_info;

      std::pair<float, float> e_force_pair = _fft->get_electro_force(grid->get_grid_idx(), grid->get_row_idx());
      electro_info.electro_force_x = e_force_pair.first;
      electro_info.electro_force_y = e_force_pair.second;

      float electro_phi = _fft->get_electro_phi(grid->get_grid_idx(), grid->get_row_idx());
      electro_info.electro_phi = electro_phi;

      _sum_phi += electro_phi * static_cast<float>(grid->get_occupied_area());

      auto it = _electro_map.find(grid);
      if (it != _electro_map.end()) {
        it->second = std::move(electro_info);
      }
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

    ElectroInfo electro_info = _electro_map.find(grid)->second;
    gradient_x += overlap_area * electro_info.electro_force_x;
    gradient_y += overlap_area * electro_info.electro_force_y;
  }

  return Point<float>(gradient_x, gradient_y);
}

void ElectricFieldGradient::reset()
{
  _sum_phi = 0;

  // _electro_map.clear();
  // for (auto pair : _electro_map) {
  //   pair.second.reset();
  // }
}

}  // namespace ipl