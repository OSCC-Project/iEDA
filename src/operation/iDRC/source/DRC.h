#pragma once
#include <assert.h>

#include <any>

#include "data/basic/BoostType.h"

namespace idb {
class IdbBuilder;

}

namespace idrc {

class DrcConfig;
class DrcDesign;
class Tech;
class RoutingSpacingCheck;
class CutSpacingCheck;
class RoutingWidthCheck;
class EnclosedAreaCheck;
class RoutingAreaCheck;
class EnclosureCheck;
class NotchSpacingCheck;
class MinStepCheck;
class CornerFillSpacingCheck;
class CutEolSpacingCheck;
class JogSpacingCheck;
class DrcSpot;
class DrcRect;
class DrcPolygon;
class RegionQuery;
class IDRWrapper;
class DrcIDBWrapper;
class DrcNet;
class SpotParser;
class MultiPatterning;
class DrcConflictGraph;
class EOLSpacingCheck;
class DrcViolationSpot;

#define DrcInst (idrc::DRC::getInst())

class DRC
{
 public:
  static DRC& getInst();
  static void destroyInst();
  DRC();
  ~DRC() {}
  DRC(const DRC& in) {}

  //通过配置文件初始化iDRC
  void initDRC(std::string& drc_config_path, idb::IdbBuilder* idb_builder = nullptr);
  //通过DataManager初始化
  void initDRC();
  // check def init
  void initDesign(std::map<std::string, std::any> config_map);

  //初始化Tech中的设计规则数据，有通过配置文件和idb_builder指针两种方式
  // void initTechFromIDB(std::string& drc_config_path);
  void initTechFromIDB(idb::IdbBuilder* idb_builder);
  //与iRT进行交互接口
  std::vector<std::pair<DrcRect*, DrcRect*>> checkiRTResult(const LayerNameToRTreeMap& layer_to_rects_rtree_map);

  // void checkViolationInRRNetList(std::vector<idr::RRNet>& rr_net_list);
  //更新当前各个设计规则检查模块的过程数据与存储结果，以备下一轮设计规则检查
  void update();
  //初始化各个设计规则检查模块
  void initCheckModule();
  //运行各个设计规则检查模块
  void run();
  //以文件的形式报告设计规则违规
  void report();
  std::map<std::string, int> getDrcResult();
  std::map<std::string, std::vector<DrcViolationSpot*>> getDrcDetailResult();

  // //对目标Net进行设计规则检查
  // void checkTargetNet(int netId);

  // 读取DEF文件的模式下获取各个设计规则违规的数目
  // int getShortViolationNum();
  // int getSpacingViolationNum();
  // int getWidthViolationNum();
  // int getAreaViolationNum();
  // int getEnclosedAreaViolationNum();

  // //获取各个设计规则检查模块下的Spot列表
  // std::map<int, std::vector<DrcSpot>>& getShortSpotList();
  // std::map<int, std::vector<DrcSpot>>& getSpacingSpotList();
  // std::map<int, std::vector<DrcSpot>>& getWidthSpotList();
  // std::map<int, std::vector<DrcSpot>>& getAreaSpotList();
  // std::map<int, std::vector<DrcSpot>>& getEnclosedAreaSpotList();

  // // debug
  // void printRTree();
  // void getObjectNum();
  ///////multi patterning
  // void checkMultipatterning(int check_colorable_num);
  // getter
  DrcConfig* get_config() { return _config; }
  DrcDesign* get_drc_design() { return _drc_design; }
  RegionQuery* get_region_query() { return _region_query; }
  Tech* get_tech() { return _tech; }

  //////////////////工程上没用到////////////////////////////////////
  // init drc polygon
  // void initDesignBlockPolygon();
  // void initNetsMergePolygon();
  // void initNetMergePolygon(DrcNet* net);
  // void bindRectangleToPolygon(DrcPolygon* polygon);
  // void initNetMergePolyEdge(DrcPolygon* polygon);  // not use now
  // init conflict graph by polygon
  // void initConflictGraphByPolygon();
  ///////////////////工程上没用到///////////////////////////////////
  ////////////////////////////

 private:
  static DRC* _drc_instance;
  DrcConfig* _config;
  DrcDesign* _drc_design;
  Tech* _tech;
  IDRWrapper* _idr_wrapper;
  DrcIDBWrapper* _idb_wrapper;

  RegionQuery* _region_query;
  JogSpacingCheck* _jog_spacing_check;
  NotchSpacingCheck* _notch_spacing_check;
  MinStepCheck* _min_step_check;
  CornerFillSpacingCheck* _corner_fill_spacing_check;
  CutEolSpacingCheck* _cut_eol_spacing_check;
  RoutingSpacingCheck* _routing_sapcing_check;
  EOLSpacingCheck* _eol_spacing_check;
  RoutingAreaCheck* _routing_area_check;
  RoutingWidthCheck* _routing_width_check;
  CutSpacingCheck* _cut_spacing_check;
  EnclosedAreaCheck* _enclosed_area_check;
  EnclosureCheck* _enclosure_check;
  SpotParser* _spot_parser;

  DrcConflictGraph* _conflict_graph;
  MultiPatterning* _multi_patterning;

  // function

  void clearRoutingShapesInDrcNetList();

  // void addSegmentToDrcPolygon(const BoostSegment& segment, DrcPolygon* polygon);
  // void initNetMergePolyEdgeOuter(DrcPolygon* polygon, std::set<int>& x_value_list, std::set<int>& y_value_list);
  // void initNetMergePolyEdgeInner(DrcPolygon* polygon, std::set<int>& x_value_list, std::set<int>& y_value_list);
};
}  // namespace idrc
