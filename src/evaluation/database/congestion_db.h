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

struct CongestionValue {
    double max_congestion;
    double total_congestion;
};

struct CongestionPin
{
  int32_t lx;
  int32_t ly;
};

struct CongestionNet
{
  std::string name;
  std::vector<CongestionPin> pins;
};

struct CongestionRegion
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
};

using CongestionNets = std::vector<CongestionNet>;

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

}  // namespace ieval