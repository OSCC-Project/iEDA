/*
 * @FilePath: density_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "density_db.h"
namespace ieval {

struct MarginGrid
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
  int32_t margin;
};

class DensityEval
{
 public:
  DensityEval();
  ~DensityEval();
  static DensityEval* getInst();
  static void destroyInst();

  std::string evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);
  std::string evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);
  std::string evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);

  std::string evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);

  std::string evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);

  std::string evalHorizonMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size);
  std::string evalVerticalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size);
  std::string evalAllMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size);

  void initIDB();
  void destroyIDB();
  void initIDBCells();
  void initIDBNets();
  void initIDBRegion();

  DensityCells getDensityCells();
  DensityPins getDensityPins();
  DensityNets getDensityNets();
  DensityRegion getDensityRegion();
  DensityRegion getDensityRegionCore();

  int32_t getRowHeight();

 private:
  static DensityEval* _density_eval;

  std::string evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type, std::string output_filename);
  std::string evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, bool neighbor, std::string pin_type,
                             std::string output_filename);
  std::string evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, bool neighbor, std::string net_type,
                             std::string output_filename);
  std::string evalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string margin_type,
                         std::string output_filename);
  std::vector<MarginGrid> initMarginGrid(DensityRegion die, int32_t grid_size);
};

}  // namespace ieval