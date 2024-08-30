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
  overflow_summary.total_overflow = congestion_eval.evalTotalOverflow(map_path);
  overflow_summary.max_overflow = congestion_eval.evalMaxOverflow(map_path);
  overflow_summary.weighted_average_overflow = congestion_eval.evalAvgOverflow(map_path);

  return overflow_summary;
}

UtilzationSummary CongestionAPI::rudyUtilzation(std::string map_path)
{
  UtilzationSummary utilzation_summary;

  CongestionEval congestion_eval;
  utilzation_summary.max_utilization = congestion_eval.evalMaxUtilization(map_path);
  utilzation_summary.weighted_average_utilization = congestion_eval.evalAvgUtilization(map_path);

  return utilzation_summary;
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
