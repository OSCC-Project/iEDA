#include "../../utility/plot/PyPlot.hh"
#include "SAEvaluateWl.hh"
#include "SAPlacementPlot.hh"

namespace imp {

template <typename CoordType, typename RepresentType>
bool SAPlacementPlot<CoordType, RepresentType>::operator()(const std::string& filename, SAPlacement<CoordType, RepresentType>& placement)
{
  PyPlot<int64_t> plt;
  // assume macros are at start of instance list
  const auto& dx = *(placement.get_dx());
  const auto& dy = *(placement.get_dy());
  const auto region_lx = placement.get_region_lx();
  const auto region_ly = placement.get_region_ly();
  const auto region_dx = placement.get_region_dx();
  const auto region_dy = placement.get_region_dy();
  const auto num_moveable = placement.get_num_moveable();
  const auto num_macros = num_moveable;

  std::vector<CoordType> lx = *(placement.get_initial_lx());
  std::vector<CoordType> ly = *(placement.get_initial_ly());
  placement.packing(lx, ly);

  for (size_t i = 0; i < num_macros; ++i) {
    plt.addMacro(lx[i], ly[i], dx[i], dy[i]);
  }
  // for (size_t i = num_macros; i < num_moveable; ++i) {
  //   plt.addCluster(lx[i], ly[i], dx[i], dy[i]);
  // }
  plt.addRectangle(region_lx, region_ly, region_dx, region_dy);
  plt.set_limitation(int(region_dx * 1.1), int(region_dy * 1.1));
  return plt.save(filename);
}

}  // namespace imp