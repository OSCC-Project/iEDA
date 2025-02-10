/*
 * @FilePath: feature_eval_congestion.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-06 11:00:28
 * @Description:
 */

#include "congestion_api.h"
#include "feature_builder.h"

namespace ieda_feature {

CongestionSummary FeatureBuilder::buildCongestionEvalSummary(int32_t grid_size)
{
  CongestionSummary congestion_summary;

  //   ieval::EGRMapSummary eval_egr_map_summary = CONGESTION_API_INST->egrMap();
  //   congestion_summary.egr_map_summary.horizontal_sum = eval_egr_map_summary.horizontal_sum;
  //   congestion_summary.egr_map_summary.vertical_sum = eval_egr_map_summary.vertical_sum;
  //   congestion_summary.egr_map_summary.union_sum = eval_egr_map_summary.union_sum;

  //   ieval::RUDYMapSummary eval_rudy_map_summary = CONGESTION_API_INST->rudyMap(grid_size);
  //   congestion_summary.rudy_map_summary.rudy_horizontal = eval_rudy_map_summary.rudy_horizontal;
  //   congestion_summary.rudy_map_summary.rudy_vertical = eval_rudy_map_summary.rudy_vertical;
  //   congestion_summary.rudy_map_summary.rudy_union = eval_rudy_map_summary.rudy_union;
  //   congestion_summary.rudy_map_summary.lutrudy_horizontal = eval_rudy_map_summary.lutrudy_horizontal;
  //   congestion_summary.rudy_map_summary.lutrudy_vertical = eval_rudy_map_summary.lutrudy_vertical;
  //   congestion_summary.rudy_map_summary.lutrudy_union = eval_rudy_map_summary.lutrudy_union;

  //   ieval::OverflowSummary eval_overflow_summary = CONGESTION_API_INST->egrOverflow();
  //   congestion_summary.overflow_summary.total_overflow_horizontal = eval_overflow_summary.total_overflow_horizontal;
  //   congestion_summary.overflow_summary.total_overflow_vertical = eval_overflow_summary.total_overflow_vertical;
  //   congestion_summary.overflow_summary.total_overflow_union = eval_overflow_summary.total_overflow_union;
  //   congestion_summary.overflow_summary.max_overflow_horizontal = eval_overflow_summary.max_overflow_horizontal;
  //   congestion_summary.overflow_summary.max_overflow_vertical = eval_overflow_summary.max_overflow_vertical;
  //   congestion_summary.overflow_summary.max_overflow_union = eval_overflow_summary.max_overflow_union;
  //   congestion_summary.overflow_summary.weighted_average_overflow_horizontal =
  //   eval_overflow_summary.weighted_average_overflow_horizontal; congestion_summary.overflow_summary.weighted_average_overflow_vertical =
  //   eval_overflow_summary.weighted_average_overflow_vertical; congestion_summary.overflow_summary.weighted_average_overflow_union =
  //   eval_overflow_summary.weighted_average_overflow_union;

  //   ieval::UtilizationSummary eval_utilization_summary = CONGESTION_API_INST->rudyUtilization(false);
  //   congestion_summary.rudy_utilization_summary.max_utilization_horizontal = eval_utilization_summary.max_utilization_horizontal;
  //   congestion_summary.rudy_utilization_summary.max_utilization_vertical = eval_utilization_summary.max_utilization_vertical;
  //   congestion_summary.rudy_utilization_summary.max_utilization_union = eval_utilization_summary.max_utilization_union;
  //   congestion_summary.rudy_utilization_summary.weighted_average_utilization_horizontal
  //       = eval_utilization_summary.weighted_average_utilization_horizontal;
  //   congestion_summary.rudy_utilization_summary.weighted_average_utilization_vertical
  //       = eval_utilization_summary.weighted_average_utilization_vertical;
  //   congestion_summary.rudy_utilization_summary.weighted_average_utilization_union
  //       = eval_utilization_summary.weighted_average_utilization_union;

  //   ieval::UtilizationSummary eval_lut_utilization_summary = CONGESTION_API_INST->rudyUtilization(true);
  //   congestion_summary.lutrudy_utilization_summary.max_utilization_horizontal = eval_lut_utilization_summary.max_utilization_horizontal;
  //   congestion_summary.lutrudy_utilization_summary.max_utilization_vertical = eval_lut_utilization_summary.max_utilization_vertical;
  //   congestion_summary.lutrudy_utilization_summary.max_utilization_union = eval_lut_utilization_summary.max_utilization_union;
  //   congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_horizontal
  //       = eval_lut_utilization_summary.weighted_average_utilization_horizontal;
  //   congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_vertical
  //       = eval_lut_utilization_summary.weighted_average_utilization_vertical;
  //   congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_union
  //       = eval_lut_utilization_summary.weighted_average_utilization_union;

  return congestion_summary;
}

}  // namespace ieda_feature