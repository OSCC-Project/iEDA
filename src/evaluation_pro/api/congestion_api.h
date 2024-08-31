/*
 * @FilePath: congestion_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

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
  UtilizationSummary rudyUtilization(std::string map_path, bool use_lut = false);

  EGRReportSummary egrReport(float threshold);
};

}  // namespace ieval