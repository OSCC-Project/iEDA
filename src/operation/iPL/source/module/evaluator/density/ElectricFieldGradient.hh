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

#include "DensityGradient.hh"
#include "fft/fft.h"

namespace ipl {

struct ElectroInfo
{
  ElectroInfo()                   = default;
  ElectroInfo(const ElectroInfo&) = default;
  ElectroInfo(ElectroInfo&& other)
  {
    electro_phi     = other.electro_phi;
    electro_force_x = other.electro_force_x;
    electro_force_y = other.electro_force_y;
    other.reset();
  }

  ElectroInfo& operator=(const ElectroInfo&) = default;

  ElectroInfo& operator=(ElectroInfo&& other)
  {
    electro_phi     = other.electro_phi;
    electro_force_x = other.electro_force_x;
    electro_force_y = other.electro_force_y;
    other.reset();
    return (*this);
  }

  void reset();

  float electro_phi     = 0.0F;
  float electro_force_x = 0.0F;
  float electro_force_y = 0.0F;
};
inline void ElectroInfo::reset()
{
  electro_phi     = 0.0F;
  electro_force_x = 0.0F;
  electro_force_y = 0.0F;
}

class ElectricFieldGradient : public DensityGradient
{
 public:
  ElectricFieldGradient() = delete;
  explicit ElectricFieldGradient(GridManager* grid_manager);
  ElectricFieldGradient(const ElectricFieldGradient&) = delete;
  ElectricFieldGradient(ElectricFieldGradient&&)      = delete;
  ~ElectricFieldGradient() override                   = default;

  ElectricFieldGradient& operator=(const ElectricFieldGradient&) = delete;
  ElectricFieldGradient& operator=(ElectricFieldGradient&&) = delete;

  void         updateDensityForce(int32_t thread_num) override;
  Point<float> obtainDensityGradient(Rectangle<int32_t> shape, float scale) override;

  void reset();

 private:
  float _sum_phi;
  FFT*  _fft;

  std::unordered_map<Grid*, ElectroInfo> _electro_map;
  void                                   initElectroMap();
};
inline ElectricFieldGradient::ElectricFieldGradient(GridManager* grid_manager) : DensityGradient(grid_manager), _sum_phi(0.0F)
{
  int32_t grid_size_x = grid_manager->get_grid_size_x();
  int32_t grid_size_y = grid_manager->get_grid_size_y();

  _fft = new FFT(_grid_manager->obtainGridCntX(), _grid_manager->obtainRowCntY(), grid_size_x, grid_size_y);

  initElectroMap();
}

}  // namespace ipl

#endif