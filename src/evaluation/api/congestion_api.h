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

  EGRMapSummary egrMap(std::string stage);
  EGRMapSummary egrMapPure(std::string stage);
  OverflowSummary egrOverflow(std::string stage);
  RUDYMapSummary rudyMap(std::string stage, int32_t grid_size = 1);
  RUDYMapSummary rudyMapPure(std::string stage, int32_t grid_size = 1);
  UtilizationSummary rudyUtilization(std::string stage, bool use_lut = false);

  EGRMapSummary egrMap(std::string stage, std::string rt_dir_path);
  EGRMapSummary egrMapPure(std::string stage, std::string rt_dir_path);
  OverflowSummary egrOverflow(std::string stage, std::string rt_dir_path);
  RUDYMapSummary rudyMap(std::string stage, CongestionNets congestion_nets, CongestionRegion region, int32_t grid_size);
  UtilizationSummary rudyUtilization(std::string stage, std::string rudy_dir_path, bool use_lut = false);

  void evalNetInfo();
  void evalNetInfoPure();
  int findPinNumber(std::string net_name);
  int findAspectRatio(std::string net_name);
  float findLness(std::string net_name);

  std::string egrUnionMap(std::string stage, std::string rt_dir_path);

 private:
  static CongestionAPI* _congestion_api_inst;
};

}  // namespace ieval