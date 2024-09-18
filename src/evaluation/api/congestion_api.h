/*
 * @FilePath: congestion_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "congestion_db.h"

namespace ieval {

#define CONGESTION_API_INST (ieval::CongestionAPI::getInst())

class CongestionAPI
{
 public:
  CongestionAPI();
  ~CongestionAPI();
  static CongestionAPI* getInst();
  static void destroyInst();

  EGRMapSummary egrMap();
  OverflowSummary egrOverflow();
  RUDYMapSummary rudyMap(int32_t grid_size = 1);
  UtilizationSummary rudyUtilization(bool use_lut = false);

  EGRMapSummary egrMap(std::string rt_dir_path);
  OverflowSummary egrOverflow(std::string rt_dir_path);
  RUDYMapSummary rudyMap(CongestionNets congestion_nets, CongestionRegion region, int32_t grid_size);
  UtilizationSummary rudyUtilization(std::string rudy_dir_path, bool use_lut = false);

  void evalNetInfo();
  int findPinNumber(std::string net_name);
  int findAspectRatio(std::string net_name);
  float findLness(std::string net_name);

 private:
  static CongestionAPI* _congestion_api_inst;
};

}  // namespace ieval