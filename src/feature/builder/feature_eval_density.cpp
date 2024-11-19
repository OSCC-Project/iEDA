/*
 * @FilePath: feature_eval_density.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-06 11:00:20
 * @Description:
 */

#include "density_api.h"
#include "feature_builder.h"

namespace ieda_feature {

DensityMapSummary FeatureBuilder::buildDensityEvalSummary(int32_t grid_size)
{
  DensityMapSummary density_map_summary;

  // ieval::DensityMapSummary eval_density_map_summary = DENSITY_API_INST->densityMap(grid_size);
  // density_map_summary.cell_map_summary.macro_density = eval_density_map_summary.cell_map_summary.macro_density;
  // density_map_summary.cell_map_summary.stdcell_density = eval_density_map_summary.cell_map_summary.stdcell_density;
  // density_map_summary.cell_map_summary.allcell_density = eval_density_map_summary.cell_map_summary.allcell_density;
  // density_map_summary.pin_map_summary.macro_pin_density = eval_density_map_summary.pin_map_summary.macro_pin_density;
  // density_map_summary.pin_map_summary.stdcell_pin_density = eval_density_map_summary.pin_map_summary.stdcell_pin_density;
  // density_map_summary.pin_map_summary.allcell_pin_density = eval_density_map_summary.pin_map_summary.allcell_pin_density;
  // density_map_summary.net_map_summary.local_net_density = eval_density_map_summary.net_map_summary.local_net_density;
  // density_map_summary.net_map_summary.global_net_density = eval_density_map_summary.net_map_summary.global_net_density;
  // density_map_summary.net_map_summary.allnet_density = eval_density_map_summary.net_map_summary.allnet_density;

  // ieval::MacroMarginSummary eval_macro_margin_summary = DENSITY_API_INST->macroMarginMap(grid_size);
  // density_map_summary.macro_margin_summary.horizontal_margin = eval_macro_margin_summary.horizontal_margin;
  // density_map_summary.macro_margin_summary.vertical_margin = eval_macro_margin_summary.vertical_margin;
  // density_map_summary.macro_margin_summary.union_margin = eval_macro_margin_summary.union_margin;

  return density_map_summary;
}

}  // namespace ieda_feature