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

DensityAPI* DensityAPI::_density_api_inst = nullptr;

DensityAPI::DensityAPI()
{
}

DensityAPI::~DensityAPI()
{
}

DensityAPI* DensityAPI::getInst()
{
  if (_density_api_inst == nullptr) {
    _density_api_inst = new DensityAPI();
  }

  return _density_api_inst;
}

void DensityAPI::destroyInst()
{
  if (_density_api_inst != nullptr) {
    delete _density_api_inst;
    _density_api_inst = nullptr;
  }
}

DensityMapSummary DensityAPI::densityMap(std::string stage, int32_t grid_size, bool neighbor)
{
  DensityMapSummary density_map_summary;

  EVAL_DENSITY_INST->initIDB();
  density_map_summary.cell_map_summary = cellDensityMap(stage, grid_size);
  density_map_summary.pin_map_summary = pinDensityMap(stage, grid_size, neighbor);
  density_map_summary.net_map_summary = netDensityMap(stage, grid_size, neighbor);
  EVAL_DENSITY_INST->destroyIDB();

  return density_map_summary;
}

DensityMapSummary DensityAPI::densityMapPure(std::string stage, int32_t grid_size, bool neighbor)
{
  DensityMapSummary density_map_summary;
  density_map_summary.cell_map_summary = cellDensityMap(stage, grid_size);
  density_map_summary.pin_map_summary = pinDensityMap(stage, grid_size, neighbor);
  density_map_summary.net_map_summary = netDensityMap(stage, grid_size, neighbor);
  return density_map_summary;
}

CellMapSummary DensityAPI::cellDensityMap(std::string stage, int32_t grid_size)
{
  CellMapSummary cell_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBCells();
  cell_map_summary = cellDensityMap(EVAL_DENSITY_INST->getDensityCells(), EVAL_DENSITY_INST->getDensityRegion(),
                                    grid_size * EVAL_DENSITY_INST->getRowHeight(), stage);

  return cell_map_summary;
}

PinMapSummary DensityAPI::pinDensityMap(std::string stage, int32_t grid_size, bool neighbor)
{
  PinMapSummary pin_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBCells();
  pin_map_summary = pinDensityMap(EVAL_DENSITY_INST->getDensityPins(), EVAL_DENSITY_INST->getDensityRegion(),
                                  grid_size * EVAL_DENSITY_INST->getRowHeight(), stage, neighbor);

  return pin_map_summary;
}

NetMapSummary DensityAPI::netDensityMap(std::string stage, int32_t grid_size, bool neighbor)
{
  NetMapSummary net_map_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBNets();
  net_map_summary = netDensityMap(EVAL_DENSITY_INST->getDensityNets(), EVAL_DENSITY_INST->getDensityRegion(),
                                  grid_size * EVAL_DENSITY_INST->getRowHeight(), stage, neighbor);

  return net_map_summary;
}

CellMapSummary DensityAPI::cellDensityMap(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage)
{
  CellMapSummary cell_map_summary;

  DensityEval density_eval;
  cell_map_summary.macro_density = density_eval.evalMacroDensity(cells, region, grid_size, stage);
  cell_map_summary.stdcell_density = density_eval.evalStdCellDensity(cells, region, grid_size, stage);
  cell_map_summary.allcell_density = density_eval.evalAllCellDensity(cells, region, grid_size, stage);

  return cell_map_summary;
}

PinMapSummary DensityAPI::pinDensityMap(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  PinMapSummary pin_map_summary;

  DensityEval density_eval;
  pin_map_summary.macro_pin_density = density_eval.evalMacroPinDensity(pins, region, grid_size, stage, neighbor);
  pin_map_summary.stdcell_pin_density = density_eval.evalStdCellPinDensity(pins, region, grid_size, stage, neighbor);
  pin_map_summary.allcell_pin_density = density_eval.evalAllCellPinDensity(pins, region, grid_size, stage, neighbor);

  return pin_map_summary;
}

NetMapSummary DensityAPI::netDensityMap(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  NetMapSummary net_map_summary;

  DensityEval density_eval;
  net_map_summary.local_net_density = density_eval.evalLocalNetDensity(nets, region, grid_size, stage, neighbor);
  net_map_summary.global_net_density = density_eval.evalGlobalNetDensity(nets, region, grid_size, stage, neighbor);
  net_map_summary.allnet_density = density_eval.evalAllNetDensity(nets, region, grid_size, stage, neighbor);

  return net_map_summary;
}

MacroCustomizedSummary DensityAPI::macroCustomizedMap(int32_t grid_size)
{
  MacroCustomizedSummary macro_customized_summary;
  std::string stage = "place";  // Default stage, can be modified as needed 

  EVAL_DENSITY_INST->initIDB();
  macro_customized_summary.margin_summary = macroMarginMap(grid_size, stage);
  macro_customized_summary.macro_channel = macroChannelMap(grid_size);
  macro_customized_summary.max_continuous_space = macroMaxContinuousSpaceMap(grid_size);
  macro_customized_summary.macro_hierarchy = macroHierarchyMap(grid_size);
  EVAL_DENSITY_INST->destroyIDB();

  return macro_customized_summary;
}

MacroMarginSummary DensityAPI::macroMarginMap(int32_t grid_size, std::string stage)
{
  MacroMarginSummary macro_margin_summary;

  EVAL_DENSITY_INST->initIDBRegion();
  EVAL_DENSITY_INST->initIDBCells();
  macro_margin_summary = macroMarginMap(EVAL_DENSITY_INST->getDensityCells(), EVAL_DENSITY_INST->getDensityRegion(),
                                        EVAL_DENSITY_INST->getDensityRegionCore(), grid_size * EVAL_DENSITY_INST->getRowHeight(), stage);

  return macro_margin_summary;
}

std::string DensityAPI::macroChannelMap(int32_t grid_size)
{
  std::string macro_channel_map;

  // EVAL_DENSITY_INST->initIDBRegion();
  // EVAL_DENSITY_INST->initIDBCells();
  // macro_channel_map = macroChannelMap(EVAL_DENSITY_INST->getDensityCells(), EVAL_DENSITY_INST->getDensityRegion(),
  //                                       grid_size * EVAL_DENSITY_INST->getRowHeight());

  return macro_channel_map;
}

std::string DensityAPI::macroMaxContinuousSpaceMap(int32_t grid_size)
{
  std::string max_continuous_space;

  return max_continuous_space;
}

std::string DensityAPI::macroHierarchyMap(int32_t grid_size)
{
  std::string macro_hierarchy;

  return macro_hierarchy;
}

MacroMarginSummary DensityAPI::macroMarginMap(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage)
{
  MacroMarginSummary macro_margin_summary;

  DensityEval density_eval;
  macro_margin_summary.horizontal_margin = density_eval.evalHorizonMargin(cells, die, core, grid_size, stage);
  macro_margin_summary.vertical_margin = density_eval.evalVerticalMargin(cells, die, core, grid_size, stage);
  macro_margin_summary.union_margin = density_eval.evalAllMargin(cells, die, core, grid_size, stage);

  return macro_margin_summary;
}

std::string DensityAPI::macroChannelMap(DensityCells cells, DensityRegion die, DensityRegion core)
{
  std::string macro_channel_map;

  return macro_channel_map;
}

std::map<int, int> DensityAPI::patchPinDensity(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, int> patch_pin_density;

  EVAL_DENSITY_INST->initIDB();

  DensityEval density_eval;
  patch_pin_density = density_eval.patchPinDensity(EVAL_DENSITY_INST->getDensityPins(), 
                                                  patch_coords);
  return patch_pin_density;
}

std::map<int, double> DensityAPI::patchCellDensity(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_cell_density;

  EVAL_DENSITY_INST->initIDB();

  DensityEval density_eval;
  patch_cell_density = density_eval.patchCellDensity(EVAL_DENSITY_INST->getDensityCells(), 
                                                  patch_coords);
  return patch_cell_density;
}


std::map<int, double> DensityAPI::patchNetDensity(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_net_density;

  EVAL_DENSITY_INST->initIDB();

  DensityEval density_eval;
  patch_net_density = density_eval.patchNetDensity(EVAL_DENSITY_INST->getDensityNets(), 
                                                  patch_coords);

  return patch_net_density;                                                
}

std::map<int, int> DensityAPI::patchMacroMargin(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, int> patch_macro_margin;

  EVAL_DENSITY_INST->initIDB();

  DensityEval density_eval;
  patch_macro_margin = density_eval.patchMacroMargin(EVAL_DENSITY_INST->getDensityCells(), EVAL_DENSITY_INST->getDensityRegionCore(), patch_coords);

  return patch_macro_margin;
}

}  // namespace ieval