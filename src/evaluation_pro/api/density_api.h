#pragma once

#include "density_db.h"

namespace ieval {
class DensityAPI
{
 public:
  DensityAPI();
  ~DensityAPI();

  CellMapSummary cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size);
  PinMapSummary pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size);
  NetMapSummary netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size);

  CellReportSummary cellDensityReport(float threshold);
  PinReportSummary pinDensityReport(float threshold);
  NetReportSummary netDensityReport(float threshold);
};

}  // namespace ieval