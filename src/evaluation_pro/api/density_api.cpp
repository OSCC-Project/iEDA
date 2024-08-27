#include "density_api.h"

#include "density_eval.h"

namespace ieval {

DensityAPI::DensityAPI()
{
}

DensityAPI::~DensityAPI()
{
}

DensityMapPathSummary DensityAPI::getMapPathSummary()
{
  DensityMapPathSummary map_path_summary;

  DensityEval density_eval;
  map_path_summary.cell_density = density_eval.evalCellDensity();
  map_path_summary.pin_density = density_eval.evalPinDensity();
  map_path_summary.net_density = density_eval.evalNetDensity();
  map_path_summary.channel_density = density_eval.evalChannelDensity();
  map_path_summary.whitespace_density = density_eval.evalWhitespaceDensity();

  return map_path_summary;
}

}  // namespace ieval