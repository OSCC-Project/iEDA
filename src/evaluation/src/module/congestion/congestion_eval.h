/*
 * @FilePath: congestion_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include <map>
#include <tuple>
#include <unordered_map>

#include "congestion_db.h"

namespace ieval {

using namespace ::std;

struct NetMetadata
{
  int32_t lx, ly, ux, uy;     // 预计算的net边界框
  double hor_rudy, ver_rudy;  // 预计算的RUDY因子
};

struct CongestionPairHash
{
  template <typename T1, typename T2>
  size_t operator()(const std::pair<T1, T2>& p) const
  {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ (hash2 << 1);
  }
};

class CongestionEval
{
 public:
  CongestionEval();
  ~CongestionEval();
  static CongestionEval* getInst();
  static void destroyInst();

  string evalHoriEGR(string stage, string rt_dir_path);
  string evalVertiEGR(string stage, string rt_dir_path);
  string evalUnionEGR(string stage, string rt_dir_path);

  string evalHoriRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalVertiRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalUnionRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);

  string evalHoriLUTRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalVertiLUTRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);
  string evalUnionLUTRUDY(string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size);

  int32_t evalHoriTotalOverflow(string stage, string rt_dir_path);
  int32_t evalVertiTotalOverflow(string stage, string rt_dir_path);
  int32_t evalUnionTotalOverflow(string stage, string rt_dir_path);

  int32_t evalHoriMaxOverflow(string stage, string rt_dir_path);
  int32_t evalVertiMaxOverflow(string stage, string rt_dir_path);
  int32_t evalUnionMaxOverflow(string stage, string rt_dir_path);

  float evalHoriAvgOverflow(string stage, string rt_dir_path);
  float evalVertiAvgOverflow(string stage, string rt_dir_path);
  float evalUnionAvgOverflow(string stage, string rt_dir_path);

  float evalHoriMaxUtilization(string stage, string rudy_dir_path, bool use_lut = false);
  float evalVertiMaxUtilization(string stage, string rudy_dir_path, bool use_lut = false);
  float evalUnionMaxUtilization(string stage, string rudy_dir_path, bool use_lut = false);

  float evalHoriAvgUtilization(string stage, string rudy_dir_path, bool use_lut = false);
  float evalVertiAvgUtilization(string stage, string rudy_dir_path, bool use_lut = false);
  float evalUnionAvgUtilization(string stage, string rudy_dir_path, bool use_lut = false);

  void initEGR();
  void destroyEGR();
  void initIDB();
  void destroyIDB();

  CongestionNets getCongestionNets();
  CongestionRegion getCongestionRegion();

  CongestionValue calRUDY(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
  CongestionValue calLUTRUDY(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
  CongestionValue calEGRCongestion(const std::string& save_path = "");

  void evalNetInfo();
  int findPinNumber(std::string net_name);
  int findAspectRatio(std::string net_name);
  double findLness(std::string net_name);
  int32_t findBBoxWidth(std::string net_name);
  int32_t findBBoxHeight(std::string net_name);
  int64_t findBBoxArea(std::string net_name);
  int32_t findBBoxLx(std::string net_name);
  int32_t findBBoxLy(std::string net_name);
  int32_t findBBoxUx(std::string net_name);
  int32_t findBBoxUy(std::string net_name);
  double findXEntropy(std::string net_name);
  double findYEntropy(std::string net_name);
  double findAvgXNNDistance(std::string net_name);
  double findStdXNNDistance(std::string net_name);
  double findRatioXNNDistance(std::string net_name);
  double findAvgYNNDistance(std::string net_name);
  double findStdYNNDistance(std::string net_name);
  double findRatioYNNDistance(std::string net_name);

  int32_t getRowHeight();
  std::string getEGRDirPath();
  std::string getDefaultOutputDir();
  void setEGRDirPath(std::string egr_dir_path);

  std::map<std::string, std::vector<std::vector<int>>> getEGRMap(bool is_run_egr = true);
  std::map<std::string, std::vector<std::vector<int>>> getDemandSupplyDiffMap(bool is_run_egr = true);
  std::map<int, double> patchRUDYCongestion(CongestionNets nets,
                                            std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);
  std::map<int, double> patchEGRCongestion(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);
  std::map<int, std::map<std::string, double>> patchLayerEGRCongestion(
      std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);

 private:
  static CongestionEval* _congestion_eval;

  std::map<std::string, int> _name_pin_numer;
  std::map<std::string, int> _name_aspect_ratio;
  std::map<std::string, double> _name_lness;
  std::map<std::string, int32_t> _name_bbox_width;
  std::map<std::string, int32_t> _name_bbox_height;
  std::map<std::string, int64_t> _name_bbox_area;
  std::map<std::string, int32_t> _name_bbox_lx;
  std::map<std::string, int32_t> _name_bbox_ly;
  std::map<std::string, int32_t> _name_bbox_ux;
  std::map<std::string, int32_t> _name_bbox_uy;
  std::map<std::string, double> _name_x_entropy;
  std::map<std::string, double> _name_y_entropy;
  std::map<std::string, double> _name_avg_x_nn_distance;
  std::map<std::string, double> _name_std_x_nn_distance;
  std::map<std::string, double> _name_ratio_x_nn_distance;
  std::map<std::string, double> _name_avg_y_nn_distance;
  std::map<std::string, double> _name_std_y_nn_distance;
  std::map<std::string, double> _name_ratio_y_nn_distance;

  string evalEGR(string rt_dir_path, string egr_type, string output_filename);
  string evalRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string rudy_type, string output_filename);
  string evalLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string lutrudy_type, string output_filename);
  double calculateLness(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t net_lx, int32_t net_ux, int32_t net_ly, int32_t net_uy);
  int32_t calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_min);
  int32_t calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_min);
  int32_t calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_max);
  int32_t calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_max);
  double getLUT(int32_t pin_num, int32_t aspect_ratio, double l_ness);
  int32_t evalTotalOverflow(string stage, string rt_dir_path, string overflow_type);
  int32_t evalMaxOverflow(string stage, string rt_dir_path, string overflow_type);
  float evalAvgOverflow(string stage, string rt_dir_path, string overflow_type);
  float evalMaxUtilization(string stage, string rudy_dir_path, string utilization_type, bool use_lut = false);
  float evalAvgUtilization(string stage, string rudy_dir_path, string utilization_type, bool use_lut = false);
  std::vector<NetMetadata> precomputeNetData(const CongestionNets& nets);

  double calculateEntropy(const std::vector<int32_t>& coords, int bin_count);
  std::tuple<double, double, double> calculateNearestNeighborStats(const std::vector<int32_t>& coords);
};
}  // namespace ieval