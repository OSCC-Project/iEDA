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

  void parseGuideFile(const std::string& guide_path);
  double parseEGRWL(std::string guide_path);
  float parseNetEGRWL(std::string guide_path, std::string net_name);
  float parsePathEGRWL(std::string guide_path, std::string net_name, std::string load_name);

  std::unordered_map<std::string, LayerDirection> parseLayerDirection(std::string guide_path);

 private:
  static InitEGR* _init_egr;
  std::string _egr_dir_path;
  std::unordered_map<std::string, float> _net_lengths;
};

}  // namespace ieval
