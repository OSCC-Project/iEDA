/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 17:30:51
 * @LastEditTime: 2023-03-03 20:14:08
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/evaluator/density/Density.cc
 * Contact : https://github.com/sjchanson
 */

#include "Density.hh"

namespace ipl {

int64_t Density::obtainOverflowArea()
{
  int64_t overflow_area = 0;
  for (auto* row : _grid_manager->get_row_list()) {
    for (auto* grid : row->get_grid_list()) {
      int64_t overflow = static_cast<int64_t>(grid->obtainGridOverflowArea());
      if (overflow > 0) {
        overflow_area += overflow;
      }
    }
  }

  return overflow_area;
}

std::vector<Grid*> Density::obtainOverflowIllegalGridList()
{
  return _grid_manager->obtainOverflowIllegalGridList();
}

float Density::obtainPeakBinDensity(){
  float peak_density = __FLT_MIN__;

  for(auto* grid_row : _grid_manager->get_row_list()){
    for(auto* grid : grid_row->get_grid_list()){
      double grid_density = grid->obtainGridDensity() / grid->get_available_ratio();
      if(grid_density > peak_density){
        peak_density = grid_density;
      }
    }
  }

  return peak_density;
}

}  // namespace ipl