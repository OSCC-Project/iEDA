#pragma once

#include "congestion_db.h"

namespace ieval {

class CongestionAPI
{
 public:
  CongestionAPI();
  ~CongestionAPI();

  EGRMapSummary egrMap(std::string map_path);
  RUDYMapSummary rudyMap(CongestionNets congestion_nets, CongestionRegion region, int32_t grid_size);

  OverflowSummary egrOverflow(std::string map_path);
  UtilzationSummary rudyUtilzation(std::string map_path);

  EGRReportSummary egrReport(float threshold);
};

}  // namespace ieval