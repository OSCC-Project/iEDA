/*
 * @FilePath: init_egr.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include <string>
#include <unordered_map>

namespace ieval {

enum class LayerDirection
{
  Horizontal,
  Vertical
};

class InitEGR
{
 public:
  InitEGR();
  ~InitEGR();
  static InitEGR* getInst();
  static void destroyInst();

  void runEGR(bool enable_timing = false);
  std::string getEGRDirPath() { return _egr_dir_path; }
  void setEGRDirPath(std::string egr_dir_path) { _egr_dir_path = egr_dir_path; }
  float getNetEGRWL(std::string net_name);
  void setBottomRoutingLayer(const std::string& layer) { _bottom_layer_override = layer; }
  void setTopRoutingLayer(const std::string& layer) { _top_layer_override = layer; }
  void setEnableTimingOverride(bool enable) { _enable_timing_override_set = true; _enable_timing_override = enable; }
  void setThreadNumberOverride(int threads) { _thread_number_override_set = true; _thread_number_override = threads; }
  void setOutputInterResultOverride(int flag) { _output_inter_result_override_set = true; _output_inter_result_override = flag; }
  void setStage(const std::string& stage) { _stage_override = stage; }
  void setResolveCongestion(const std::string& level) { _resolve_congestion_override = level; }

  void parseGuideFile(const std::string& guide_path);
  double parseEGRWL(std::string guide_path);
  float parseNetEGRWL(std::string guide_path, std::string net_name);
  float parsePathEGRWL(std::string guide_path, std::string net_name, std::string load_name);

  std::unordered_map<std::string, LayerDirection> parseLayerDirection(std::string guide_path);

 private:
  static InitEGR* _init_egr;
  std::string _egr_dir_path;
  std::unordered_map<std::string, float> _net_lengths;
  std::string _bottom_layer_override;
  std::string _top_layer_override;
  bool _enable_timing_override_set = false;
  bool _enable_timing_override = false;
  bool _thread_number_override_set = false;
  int _thread_number_override = 128;
  bool _output_inter_result_override_set = false;
  int _output_inter_result_override = 1;
  std::string _stage_override = "egr3D";
  std::string _resolve_congestion_override = "low";
};

}  // namespace ieval
