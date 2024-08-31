/*
 * @FilePath: congestion_api.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "congestion_api.h"

#include "congestion_eval.h"

namespace ieval {

CongestionAPI::CongestionAPI()
{
}

CongestionAPI::~CongestionAPI()
{
}

EGRMapSummary CongestionAPI::egrMap(std::string map_path)
{
  EGRMapSummary egr_map_summary;

  CongestionEval congestion_eval;
  egr_map_summary.horizontal_sum = congestion_eval.evalHoriEGR(map_path);
  egr_map_summary.vertical_sum = congestion_eval.evalVertiEGR(map_path);
  egr_map_summary.union_sum = congestion_eval.evalUnionEGR(map_path);

  return egr_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMap(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;

  CongestionEval congestion_eval;
  rudy_map_summary.rudy_horizontal = congestion_eval.evalHoriRUDY(nets, region, grid_size);
  rudy_map_summary.rudy_vertical = congestion_eval.evalVertiRUDY(nets, region, grid_size);
  rudy_map_summary.rudy_union = congestion_eval.evalUnionRUDY(nets, region, grid_size);

  rudy_map_summary.lutrudy_horizontal = congestion_eval.evalHoriLUTRUDY(nets, region, grid_size);
  rudy_map_summary.lutrudy_vertical = congestion_eval.evalVertiLUTRUDY(nets, region, grid_size);
  rudy_map_summary.lutrudy_union = congestion_eval.evalUnionLUTRUDY(nets, region, grid_size);

  return rudy_map_summary;
}

OverflowSummary CongestionAPI::egrOverflow(std::string map_path)
{
  OverflowSummary overflow_summary;

  CongestionEval congestion_eval;
  overflow_summary.total_overflow_horizontal = congestion_eval.evalHoriTotalOverflow(map_path);
  overflow_summary.total_overflow_vertical = congestion_eval.evalVertiTotalOverflow(map_path);
  overflow_summary.total_overflow_union = congestion_eval.evalUnionTotalOverflow(map_path);

  overflow_summary.max_overflow_horizontal = congestion_eval.evalHoriMaxOverflow(map_path);
  overflow_summary.max_overflow_vertical = congestion_eval.evalVertiMaxOverflow(map_path);
  overflow_summary.max_overflow_union = congestion_eval.evalUnionMaxOverflow(map_path);

  overflow_summary.weighted_average_overflow_horizontal = congestion_eval.evalHoriAvgOverflow(map_path);
  overflow_summary.weighted_average_overflow_vertical = congestion_eval.evalVertiAvgOverflow(map_path);
  overflow_summary.weighted_average_overflow_union = congestion_eval.evalUnionAvgOverflow(map_path);

  return overflow_summary;
}

UtilizationSummary CongestionAPI::rudyUtilization(std::string map_path, bool use_lut)
{
  UtilizationSummary utilization_summary;

  CongestionEval congestion_eval;
  utilization_summary.max_utilization_horizontal = congestion_eval.evalHoriMaxUtilization(map_path, use_lut);
  utilization_summary.max_utilization_vertical = congestion_eval.evalVertiMaxUtilization(map_path, use_lut);
  utilization_summary.max_utilization_union = congestion_eval.evalUnionMaxUtilization(map_path, use_lut);

  utilization_summary.weighted_average_utilization_horizontal = congestion_eval.evalHoriAvgUtilization(map_path, use_lut);
  utilization_summary.weighted_average_utilization_vertical = congestion_eval.evalVertiAvgUtilization(map_path, use_lut);
  utilization_summary.weighted_average_utilization_union = congestion_eval.evalUnionAvgUtilization(map_path, use_lut);

  return utilization_summary;
}

EGRReportSummary CongestionAPI::egrReport(float threshold)
{
  EGRReportSummary egr_report_summary;

  CongestionEval congestion_eval;
  egr_report_summary.hotspot = congestion_eval.reportHotspot(threshold);
  egr_report_summary.overflow = congestion_eval.reportOverflow(threshold);

  return egr_report_summary;
}

}  // namespace ieval
