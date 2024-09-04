/*
 * @FilePath: density_api.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "density_api.h"

#include "density_eval.h"

namespace ieval {

#define EVAL_DENSITY_INST (ieval::DensityEval::getInst())

DensityAPI::DensityAPI()
{
}

DensityAPI::~DensityAPI()
{
}

DensityMapSummary DensityAPI::densityMap(int32_t grid_size)
{
  DensityMapSummary density_map_summary;

  EVAL_DENSITY_INST->initIDB();
  density_map_summary.cell_map_summary = cellDensityMap(grid_size);
  density_map_summary.pin_map_summary = pinDensityMap(grid_size);
  density_map_summary.net_map_summary = netDensityMap(grid_size);
  EVAL_DENSITY_INST->destroyIDB();

  return density_map_summary;
}

CellMapSummary DensityAPI::cellDensityMap(int32_t grid_size)
{
  CellMapSummary cell_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBCells();
  cell_map_summary = cellDensityMap(EVAL_DENSITY_INST->getDensityCells(), EVAL_DENSITY_INST->getDensityRegion(), grid_size);

  return cell_map_summary;
}

PinMapSummary DensityAPI::pinDensityMap(int32_t grid_size)
{
  PinMapSummary pin_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBCells();
  pin_map_summary = pinDensityMap(EVAL_DENSITY_INST->getDensityPins(), EVAL_DENSITY_INST->getDensityRegion(), grid_size);

  return pin_map_summary;
}

NetMapSummary DensityAPI::netDensityMap(int32_t grid_size)
{
  NetMapSummary net_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBNets();
  net_map_summary = netDensityMap(EVAL_DENSITY_INST->getDensityNets(), EVAL_DENSITY_INST->getDensityRegion(), grid_size);

  return net_map_summary;
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

}  // namespace ieval