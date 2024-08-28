#include "density_api.h"

#include "density_eval.h"

namespace ieval {

DensityAPI::DensityAPI()
{
}

DensityAPI::~DensityAPI()
{
}

CellMapSummary DensityAPI::cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size)
{
  CellMapSummary cell_map_summary;

  DensityEval density_eval;
  cell_map_summary.macro_density = density_eval.evalMacroDensity(cells, region, grid_size);
  cell_map_summary.stdcell_density = density_eval.evalStdCellDensity(cells, region, grid_size);
  cell_map_summary.allcell_density = density_eval.evalAllCellDensity(cells, region, grid_size);

  return cell_map_summary;
}

PinMapSummary DensityAPI::pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size)
{
  PinMapSummary pin_map_summary;

  DensityEval density_eval;
  pin_map_summary.macro_pin_density = density_eval.evalMacroPinDensity(pins, region, grid_size);
  pin_map_summary.stdcell_pin_density = density_eval.evalStdCellPinDensity(pins, region, grid_size);
  pin_map_summary.allcell_pin_density = density_eval.evalAllCellPinDensity(pins, region, grid_size);

  return pin_map_summary;
}

NetMapSummary DensityAPI::netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size)
{
  NetMapSummary net_map_summary;

  DensityEval density_eval;
  net_map_summary.local_net_density = density_eval.evalLocalNetDensity(nets, region, grid_size);
  net_map_summary.global_net_density = density_eval.evalGlobalNetDensity(nets, region, grid_size);
  net_map_summary.allnet_density = density_eval.evalAllNetDensity(nets, region, grid_size);

  return net_map_summary;
}

CellReportSummary DensityAPI::cellDensityReport(int32_t threshold)
{
  CellReportSummary cell_report_summary;

  DensityEval density_eval;
  cell_report_summary.macro_density = density_eval.reportMacroDensity(threshold);
  cell_report_summary.stdcell_density = density_eval.reportStdCellDensity(threshold);
  cell_report_summary.allcell_density = density_eval.reportAllCellDensity(threshold);

  return cell_report_summary;
}

PinReportSummary DensityAPI::pinDensityReport(int32_t threshold)
{
  PinReportSummary pin_report_summary;

  DensityEval density_eval;
  pin_report_summary.macro_pin_density = density_eval.reportMacroPinDensity(threshold);
  pin_report_summary.stdcell_pin_density = density_eval.reportStdCellPinDensity(threshold);
  pin_report_summary.allcell_pin_density = density_eval.reportAllCellPinDensity(threshold);

  return pin_report_summary;
}

NetReportSummary DensityAPI::netDensityReport(int32_t threshold)
{
  NetReportSummary net_report_summary;

  DensityEval density_eval;
  net_report_summary.local_net_density = density_eval.reportLocalNetDensity(threshold);
  net_report_summary.global_net_density = density_eval.reportGlobalNetDensity(threshold);
  net_report_summary.all_net_density = density_eval.reportAllNetDensity(threshold);

  return net_report_summary;
}

}  // namespace ieval