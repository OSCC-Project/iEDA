/*
 * @FilePath: congestion_db.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

using namespace ::std;

struct CongestionPin
{
  int32_t lx;
  int32_t ly;
};

struct CongestionNet
{
  vector<CongestionPin> pins;
};

struct CongestionRegion
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
};

using CongestionNets = vector<CongestionNet>;

struct EGRMapSummary
{
  string horizontal_sum;
  string vertical_sum;
  string union_sum;
};

struct RUDYMapSummary
{
  string rudy_horizontal;
  string rudy_vertical;
  string rudy_union;

  string lutrudy_horizontal;
  string lutrudy_vertical;
  string lutrudy_union;
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

}  // namespace ieval