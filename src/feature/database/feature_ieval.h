/*
 * @FilePath: feature_ieval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-04 09:52:05
 * @Description:
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "feature_ipw.h"
#include "feature_ista.h"
namespace ieda_feature {

// Wirelength
struct TotalWLSummary
{
  int64_t HPWL;
  int64_t FLUTE;
  int64_t HTree;
  int64_t VTree;
  int64_t GRWL;
};

// Density
struct CellMapSummary
{
  std::string macro_density;
  std::string stdcell_density;
  std::string allcell_density;
};

struct PinMapSummary
{
  std::string macro_pin_density;
  std::string stdcell_pin_density;
  std::string allcell_pin_density;
};

struct NetMapSummary
{
  std::string local_net_density;
  std::string global_net_density;
  std::string allnet_density;
};

struct MacroMarginSummary
{
  std::string horizontal_margin;
  std::string vertical_margin;
  std::string union_margin;
};

struct DensityMapSummary
{
  CellMapSummary cell_map_summary;
  PinMapSummary pin_map_summary;
  NetMapSummary net_map_summary;
  MacroMarginSummary macro_margin_summary;
};

// Congestion
struct EGRMapSummary
{
  std::string horizontal_sum;
  std::string vertical_sum;
  std::string union_sum;
};

struct RUDYMapSummary
{
  std::string rudy_horizontal;
  std::string rudy_vertical;
  std::string rudy_union;

  std::string lutrudy_horizontal;
  std::string lutrudy_vertical;
  std::string lutrudy_union;
};

struct OverflowSummary
{
  int32_t total_overflow_horizontal;
  int32_t total_overflow_vertical;
  int32_t total_overflow_union;

  int32_t max_overflow_horizontal;
  int32_t max_overflow_vertical;
  int32_t max_overflow_union;

  float weighted_average_overflow_horizontal;
  float weighted_average_overflow_vertical;
  float weighted_average_overflow_union;
};

struct UtilizationSummary
{
  float max_utilization_horizontal;
  float max_utilization_vertical;
  float max_utilization_union;

  float weighted_average_utilization_horizontal;
  float weighted_average_utilization_vertical;
  float weighted_average_utilization_union;
};

struct CongestionSummary
{
  EGRMapSummary egr_map_summary;
  RUDYMapSummary rudy_map_summary;
  OverflowSummary overflow_summary;
  UtilizationSummary rudy_utilization_summary;
  UtilizationSummary lutrudy_utilization_summary;
};
struct TimingEvalSummaryBase
{
  std::vector<ClockTiming> clock_timings;
  PowerInfo power_info;
};
struct WlmTimingEvalSummary : TimingEvalSummaryBase
{
};
struct HpwlTimingEvalSummary : TimingEvalSummaryBase
{
};
struct FluteTimingEvalSummary : TimingEvalSummaryBase
{
};
struct SaltTimingEvalSummary : TimingEvalSummaryBase
{
};
struct EgrTimingEvalSummary : TimingEvalSummaryBase
{
};
struct DrTimingEvalSummary : TimingEvalSummaryBase
{
};
struct TimingEvalSummary
{
  WlmTimingEvalSummary wlm_timing_eval_summary;
  HpwlTimingEvalSummary hpwl_timing_eval_summary;
  FluteTimingEvalSummary flute_timing_eval_summary;
  SaltTimingEvalSummary salt_timing_eval_summary;
  EgrTimingEvalSummary egr_timing_eval_summary;
  DrTimingEvalSummary dr_timing_eval_summary;
};

struct UnionEvalSummary
{
  TotalWLSummary total_wl_summary;
  DensityMapSummary density_map_summary;
  CongestionSummary congestion_summary;
  TimingEvalSummary timing_eval_summary;
};

}  // namespace ieda_feature
