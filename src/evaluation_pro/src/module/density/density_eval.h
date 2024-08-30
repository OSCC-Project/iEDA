
#pragma once

#include "density_db.h"

namespace ieval {
class DensityEval
{
 public:
  DensityEval();
  ~DensityEval();

  std::string evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size);
  std::string evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size);
  std::string evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size);

  std::string evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);
  std::string evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);
  std::string evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size);

  std::string evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);
  std::string evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);
  std::string evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size);

  std::string reportMacroDensity(float threshold);
  std::string reportStdCellDensity(float threshold);
  std::string reportAllCellDensity(float threshold);

  std::string reportMacroPinDensity(float threshold);
  std::string reportStdCellPinDensity(float threshold);
  std::string reportAllCellPinDensity(float threshold);

  std::string reportLocalNetDensity(float threshold);
  std::string reportGlobalNetDensity(float threshold);
  std::string reportAllNetDensity(float threshold);

 private:
  std::string evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type, std::string output_filename);
  std::string evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string pin_type, std::string output_filename);
  std::string evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string net_type, std::string output_filename);
  std::string getAbsoluteFilePath(std::string filename);
};
}  // namespace ieval