#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace idrc {

class DrcConfig;
class Tech;
class RoutingSpacingCheck;
class CutSpacingCheck;
class RoutingWidthCheck;
class EnclosedAreaCheck;
class RoutingAreaCheck;
class DrcSpot;
class DrcRect;
class EnclosureCheck;
class EOLSpacingCheck;

class SpotParser
{
 public:
  static SpotParser* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  {
    static SpotParser instance(config, tech);
    return &instance;
  }

  // 以文件的形式报告各个设计规则检查模块内部存储的违规结果
  void reportSpacingViolation(RoutingSpacingCheck* check);
  void reportCutSpacingViolation(CutSpacingCheck* check);
  void reportShortViolation(RoutingSpacingCheck* check);
  void reportWidthViolation(RoutingWidthCheck* check);
  void reportAreaViolation(RoutingAreaCheck* check);
  void reportEnclosedAreaViolation(EnclosedAreaCheck* check);
  void reportEnclosureViolation(EnclosureCheck* check);
  void reportEOLSpacingViolation(EOLSpacingCheck* check);
  void reportEnd2EndSpacingViolation(EOLSpacingCheck* check);

 private:
  DrcConfig* _config;
  Tech* _tech;

  explicit SpotParser(DrcConfig* config, Tech* tech) { init(config, tech); }
  SpotParser(const RoutingWidthCheck& other) = delete;
  SpotParser(RoutingWidthCheck&& other) = delete;
  ~SpotParser() {}
  SpotParser& operator=(const RoutingWidthCheck& other) = delete;
  SpotParser& operator=(RoutingWidthCheck&& other) = delete;
  // function
  void init(DrcConfig* config, Tech* tech);

  std::ofstream get_spot_file(std::string file_name);
  void reportSpotToFile(int layerId, DrcSpot& spot, std::ofstream& spot_file);
  void reportRectToFile(int layerId, DrcRect* rect, std::ofstream& spot_file);
  void reportTotal(std::map<int, std::vector<DrcSpot>>& layer_to_spots, std::ofstream& spot_file);
  void reportEOLSpacingSpot(int layerId, DrcSpot& spot, std::ofstream& spot_file);
  void reportEnd2EndSpacingSpot(int layerId, DrcSpot& spot, std::ofstream& spot_file);
  void reportEnd2EndSpot(int layerId, DrcSpot spot, std::ofstream& spot_file);
  void reportEOLSpot(int layerId, DrcSpot spot, std::ofstream& spot_file);
};
}  // namespace idrc
