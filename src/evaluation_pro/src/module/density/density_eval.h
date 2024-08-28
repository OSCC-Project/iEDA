
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

  std::string reportMacroDensity(int32_t threshold);
  std::string reportStdCellDensity(int32_t threshold);
  std::string reportAllCellDensity(int32_t threshold);

  std::string reportMacroPinDensity(int32_t threshold);
  std::string reportStdCellPinDensity(int32_t threshold);
  std::string reportAllCellPinDensity(int32_t threshold);

  std::string reportLocalNetDensity(int32_t threshold);
  std::string reportGlobalNetDensity(int32_t threshold);
  std::string reportAllNetDensity(int32_t threshold);

 private:
  std::string evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type, std::string output_filename);
  std::string evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string pin_type, std::string output_filename);
  std::string evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string net_type, std::string output_filename);
  std::string getAbsoluteFilePath(std::string filename);
};
}  // namespace ieval