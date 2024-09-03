/*
 * @FilePath: congestion_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "congestion_db.h"

namespace ieval {

using namespace ::std;

class CongestionEval
{
 public:
  CongestionEval();
  ~CongestionEval();
  static CongestionEval* getInst();
  static void destroyInst();

  string evalHoriEGR(string rt_dir_path);
  string evalVertiEGR(string rt_dir_path);
  string evalUnionEGR(string rt_dir_path);

  string evalHoriRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalVertiRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalUnionRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);

  string evalHoriLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalVertiLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalUnionLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size);

  int32_t evalHoriTotalOverflow(string rt_dir_path);
  int32_t evalVertiTotalOverflow(string rt_dir_path);
  int32_t evalUnionTotalOverflow(string rt_dir_path);

  int32_t evalHoriMaxOverflow(string rt_dir_path);
  int32_t evalVertiMaxOverflow(string rt_dir_path);
  int32_t evalUnionMaxOverflow(string rt_dir_path);

  float evalHoriAvgOverflow(string rt_dir_path);
  float evalVertiAvgOverflow(string rt_dir_path);
  float evalUnionAvgOverflow(string rt_dir_path);

  float evalHoriMaxUtilization(string rudy_dir_path, bool use_lut = false);
  float evalVertiMaxUtilization(string rudy_dir_path, bool use_lut = false);
  float evalUnionMaxUtilization(string rudy_dir_path, bool use_lut = false);

  float evalHoriAvgUtilization(string rudy_dir_path, bool use_lut = false);
  float evalVertiAvgUtilization(string rudy_dir_path, bool use_lut = false);
  float evalUnionAvgUtilization(string rudy_dir_path, bool use_lut = false);

  string reportHotspot(float threshold);
  string reportOverflow(float threshold);

  void initEGR();
  void destroyEGR();
  void initIDB();
  void destroyIDB();

  CongestionNets getCongestionNets();
  CongestionRegion getCongestionRegion();

 private:
  static CongestionEval* _congestion_eval;

  string evalEGR(string rt_dir_path, string egr_type, string output_filename);
  string evalRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string rudy_type, string output_filename);
  string evalLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string lutrudy_type, string output_filename);
  float calculateLness(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t net_lx, int32_t net_ux, int32_t net_ly, int32_t net_uy);
  int32_t calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_min);
  int32_t calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_min);
  int32_t calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_max);
  int32_t calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_max);
  double getLUT(int32_t pin_num, int32_t aspect_ratio, float l_ness);
  int32_t evalTotalOverflow(string rt_dir_path, string overflow_type);
  int32_t evalMaxOverflow(string rt_dir_path, string overflow_type);
  float evalAvgOverflow(string rt_dir_path, string overflow_type);
  float evalMaxUtilization(string rudy_dir_path, string utilization_type, bool use_lut = false);
  float evalAvgUtilization(string rudy_dir_path, string utilization_type, bool use_lut = false);
};
}  // namespace ieval