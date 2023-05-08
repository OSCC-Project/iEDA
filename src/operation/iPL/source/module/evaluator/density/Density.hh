/*
 * @Author: S.J Chen
 * @Date: 2022-03-07 12:08:22
 * @LastEditTime: 2023-03-03 20:13:12
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/evaluator/density/Density.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_DENSITY_H
#define IPL_EVALUATOR_DENSITY_H

#include <vector>

#include "GridManager.hh"

namespace ipl {

class Density
{
 public:
  Density() = delete;
  explicit Density(GridManager* grid_manager);
  Density(const Density&) = delete;
  Density(Density&&) = delete;
  ~Density() = default;

  Density& operator=(const Density&) = delete;
  Density& operator=(Density&&) = delete;

  int64_t obtainOverflowArea();
  std::vector<Grid*> obtainOverflowIllegalGridList();
  float obtainPeakBinDensity();

 protected:
  GridManager* _grid_manager;
};
inline Density::Density(GridManager* grid_manager) : _grid_manager(grid_manager)
{
}

}  // namespace ipl

#endif