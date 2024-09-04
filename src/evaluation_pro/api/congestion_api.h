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

  EGRMapSummary egrMap();
  OverflowSummary egrOverflow();
  RUDYMapSummary rudyMap(int32_t grid_size);
  UtilizationSummary rudyUtilization(bool use_lut = false);

  EGRMapSummary egrMap(std::string rt_dir_path);
  OverflowSummary egrOverflow(std::string rt_dir_path);
  RUDYMapSummary rudyMap(CongestionNets congestion_nets, CongestionRegion region, int32_t grid_size);
  UtilizationSummary rudyUtilization(std::string rudy_dir_path, bool use_lut = false);
};

}  // namespace ieval