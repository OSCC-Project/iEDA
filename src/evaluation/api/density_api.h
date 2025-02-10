/*
 * @FilePath: density_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "density_db.h"

#define DENSITY_API_INST (ieval::DensityAPI::getInst())

namespace ieval {
class DensityAPI
{
 public:
  DensityAPI();
  ~DensityAPI();
  static DensityAPI* getInst();
  static void destroyInst();

  DensityMapSummary densityMap(std::string stage, int32_t grid_size = 1, bool neighbor = false);
  DensityMapSummary densityMapPure(std::string stage, int32_t grid_size = 1, bool neighbor = false);
  CellMapSummary cellDensityMap(std::string stage, int32_t grid_size = 1);
  PinMapSummary pinDensityMap(std::string stage, int32_t grid_size = 1, bool neighbor = false);
  NetMapSummary netDensityMap(std::string stage, int32_t grid_size = 1, bool neighbor = false);

  CellMapSummary cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);
  PinMapSummary pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  NetMapSummary netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);

  MacroMarginSummary macroMarginMap(int32_t grid_size = 1);
  MacroMarginSummary macroMarginMap(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size);

  MacroCustomizedSummary macroCustomizedMap(int32_t grid_size = 1);
  std::string macroChannelMap(int32_t grid_size = 1);
  std::string macroChannelMap(DensityCells cells, DensityRegion die, DensityRegion core);
  std::string macroMaxContinuousSpaceMap(int32_t grid_size = 1);
  std::string macroHierarchyMap(int32_t grid_size = 1);

 private:
  static DensityAPI* _density_api_inst;
};

}  // namespace ieval