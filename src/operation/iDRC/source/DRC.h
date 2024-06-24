// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

  // Initialize iDRC using the configuration file
  void initDRC(std::string& drc_config_path, idb::IdbBuilder* idb_builder = nullptr);
  // Initialize using DataManager
  void initDRC();
  // Check DEF initialization
  void initDesign(std::map<std::string, std::any> config_map);

  // Initialize design rule data in Tech, either through the configuration file or idb_builder pointer
  // void initTechFromIDB(std::string& drc_config_path);
  void initTechFromIDB(idb::IdbBuilder* idb_builder);
  // Interface for interaction with iRT
  std::vector<std::pair<DrcRect*, DrcRect*>> checkiRTResult(const LayerNameToRTreeMap& layer_to_rects_rtree_map);

  // void checkViolationInRRNetList(std::vector<idr::RRNet>& rr_net_list);
  // Update the process data and stored results of the current design rule check modules for the next round of checks
  void update();
  // Initialize each design rule check module
  void initCheckModule();
  // Run each design rule check module
  void run();
  // Report design rule violations in a file
  void report();
  std::map<std::string, int> getDrcResult();
  std::map<std::string, std::vector<DrcViolationSpot*>> getDrcDetailResult();

  // // Perform design rule check on the target net
  // void checkTargetNet(int netId);

  // Get the number of design rule violations in DEF file mode
  // int getShortViolationNum();
  // int getSpacingViolationNum();
  // int getWidthViolationNum();
  // int getAreaViolationNum();
  // int getEnclosedAreaViolationNum();

  // // Get the list of spots under each design rule check module
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

  ////////////////// useless ///////////////////////////////////
  // init drc polygon
  // void initDesignBlockPolygon();
  // void initNetsMergePolygon();
  // void initNetMergePolygon(DrcNet* net);
  // void bindRectangleToPolygon(DrcPolygon* polygon);
  // void initNetMergePolyEdge(DrcPolygon* polygon);  // not use now
  // init conflict graph by polygon
  // void initConflictGraphByPolygon();
  /////////////////// useless  ///////////////////////////////////
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
