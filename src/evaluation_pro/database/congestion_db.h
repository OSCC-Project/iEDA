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
  int32_t total_overflow;
  int32_t max_overflow;
  float weighted_average_overflow;
};

struct UtilzationSummary
{
  float max_utilization;
  float weighted_average_utilization;
};

struct EGRReportSummary
{
  string hotspot;
  string overflow;
};

}  // namespace ieval