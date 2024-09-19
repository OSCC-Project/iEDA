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

  DensityMapSummary densityMap(int32_t grid_size = 1);
  CellMapSummary cellDensityMap(int32_t grid_size = 1);
  PinMapSummary pinDensityMap(int32_t grid_size = 1);
  NetMapSummary netDensityMap(int32_t grid_size = 1);

  CellMapSummary cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size);
  PinMapSummary pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size);
  NetMapSummary netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size);

 private:
  static DensityAPI* _density_api_inst;
};

}  // namespace ieval