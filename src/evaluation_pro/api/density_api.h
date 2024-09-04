/*
 * @FilePath: density_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "density_db.h"

namespace ieval {
class DensityAPI
{
 public:
  DensityAPI();
  ~DensityAPI();

  DensityMapSummary densityMap(int32_t grid_size);
  CellMapSummary cellDensityMap(int32_t grid_size);
  PinMapSummary pinDensityMap(int32_t grid_size);
  NetMapSummary netDensityMap(int32_t grid_size);

  CellMapSummary cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size);
  PinMapSummary pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size);
  NetMapSummary netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size);
};

}  // namespace ieval