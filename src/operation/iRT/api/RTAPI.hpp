#pragma once

#include <any>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../database/interaction/ids.hpp"

namespace irt {

#define RTAPI_INST (irt::RTAPI::getInst())

enum class Tool
{
  kDetailedRouter,
  kGlobalRouter,
  kPinAccessor,
  kResourceAllocator,
  kTrackAssigner,
  kViolationRepairer
};

class RTAPI
{
 public:
  static RTAPI& getInst();
  static void destroyInst();

  // RT
  void initRT(std::map<std::string, std::any> config_map);
  void runRT(std::vector<Tool> tool_list);
  Stage convertToStage(Tool tool);
  void destroyRT();

  // EGR
  void runEGR(std::map<std::string, std::any> config_map);

  // AI
  void runGRToAI(std::string ai_json_file_path, int lower_bound_value, int upper_bound_value);

  // EVAL
  eval::TileGrid* getCongestonMap(std::map<std::string, std::any> config_map);
  std::vector<double> getWireLengthAndViaNum(std::map<std::string, std::any> config_map);

  // DRC
  std::vector<ids::DRCRect> getMinScope(std::vector<ids::DRCRect>& detection_rect_list);

  // CTS
  std::vector<ids::PHYNode> getPHYNodeList(std::vector<ids::Segment> segment_list);

 private:
  static RTAPI* _rt_api_instance;

  RTAPI() = default;
  RTAPI(const RTAPI& other) = delete;
  RTAPI(RTAPI&& other) = delete;
  ~RTAPI() = default;
  RTAPI& operator=(const RTAPI& other) = delete;
  RTAPI& operator=(RTAPI&& other) = delete;
  // function
};

}  // namespace irt
