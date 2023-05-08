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

  virtual void updateDensityForce(int32_t thread_num) = 0;
  virtual Point<float> obtainDensityGradient(Rectangle<int32_t> shape, float scale) = 0;

 protected:
  GridManager* _grid_manager;
};
inline DensityGradient::DensityGradient(GridManager* grid_manager) : _grid_manager(grid_manager)
{
}

}  // namespace ipl

#endif