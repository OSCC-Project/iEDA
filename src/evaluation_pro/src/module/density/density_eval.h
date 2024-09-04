/*
 * @FilePath: density_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "density_db.h"
namespace ieval {
class DensityEval
{
 public:
  DensityEval();
  ~DensityEval();
  static DensityEval* getInst();
  static void destroyInst();

  std::string evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size);
  std::string evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size);
  std::string evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size);

  std::string evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);
  std::string evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);
  std::string evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);

  std::string evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);
  std::string evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);
  std::string evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);

  void initIDB();
  void destroyIDB();
  void initIDBCells();
  void initIDBNets();
  void initIDBRegion();

  DensityCells getDensityCells();
  DensityPins getDensityPins();
  DensityNets getDensityNets();
  DensityRegion getDensityRegion();

 private:
  static DensityEval* _density_eval;

  std::string evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type, std::string output_filename);
  std::string evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string pin_type, std::string output_filename);
  std::string evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string net_type, std::string output_filename);
};
}  // namespace ieval