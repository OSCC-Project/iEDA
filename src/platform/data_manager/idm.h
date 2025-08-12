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
/**
 * @File Name: idm.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../database/interaction/ids.hpp"
#include "IdbDesign.h"
#include "IdbLayout.h"
#include "builder.h"
#include "config/dm_config.h"
#include "def_service.h"
#include "lef_service.h"
#include "string/Str.hh"
#include "usage/usage.hh"

using std::string;
using std::vector;

using namespace idb;

#define dmInst idm::DataManager::getInstance()  // dmInst is DataManager*

namespace idm {

class DataManager
{
 public:
  static DataManager* getInstance()
  {
    if (!_instance) {
      _instance = new DataManager;
    }
    return _instance;
  }

  /// getter
  DataConfig& get_config() { return _config; };
  IdbBuilder* get_idb_builder() { return _idb_builder; }
  void set_idb_builder(IdbBuilder* idb_builder) { _idb_builder = idb_builder; }
  IdbDefService* get_idb_def_service() { return _idb_def_service; }
  void set_idb_def_service(IdbDefService* idb_def_service) { _idb_def_service = idb_def_service; }
  IdbLefService* get_idb_lef_service() { return _idb_lef_service; }
  void set_idb_lef_service(IdbLefService* idb_lef_service) { _idb_lef_service = idb_lef_service; }

  IdbDesign* get_idb_design() { return _idb_def_service != nullptr ? _idb_def_service->get_design() : nullptr; }
  IdbLayout* get_idb_layout() { return _idb_lef_service != nullptr ? _idb_lef_service->get_layout() : nullptr; }
  bool is_def_read() { return _idb_def_service != nullptr ? true : false; }

  int get_routing_layer_1st();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iDB init
  bool init(string config_path);
  bool readLef(string config_path);
  bool readLef(vector<string> lef_paths, bool b_techlef = false);
  bool readDef(string path);
  bool readVerilog(string path, string top_module = "");

  /// iDB save
  bool save(string name, string def_path = "");
  bool saveDef(string def_path);
  bool saveLef(string lef_path);
  void saveVerilog(string verilog_path, std::set<std::string>&& exclude_cell_names = {}, bool is_add_space_for_escape_name = false);
  bool saveGDSII(string path);
  bool saveJSON(string path, string options);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iDB layout area operation
  void initDie(int ll_x, int ll_y, int ur_x, int ur_y);
  uint64_t dieArea();
  double dieAreaUm();
  float dieUtilization();
  uint64_t coreArea();
  double coreAreaUm();
  float coreUtilization();

  /// basic config
  IdbOrient getDefaultOrient(int32_t coord_x, int32_t coord_y);
  IdbRow* createRow(string row_name, string site_name, int32_t orig_x, int32_t orig_y, IdbOrient site_orient, int32_t num_x, int32_t num_y,
                    int32_t step_x, int32_t step_y);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iDB design instance
  double instanceArea(IdbInstanceType type = IdbInstanceType::kMax);
  double distInstArea();
  double netlistInstArea();
  double timingInstArea();
  int32_t instancePinNum(string inst_name);
  IdbInstance* createInstance(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                              IdbOrient orient = IdbOrient::kN_R0, IdbInstanceType type = IdbInstanceType::kNetlist,
                              IdbPlacementStatus status = IdbPlacementStatus::kUnplaced);
  IdbInstance* createNetlistInst(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                                 IdbOrient orient = IdbOrient::kN_R0, IdbPlacementStatus status = IdbPlacementStatus::kUnplaced);
  IdbInstance* createPhysicalInst(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                                  IdbOrient orient = IdbOrient::kN_R0, IdbPlacementStatus status = IdbPlacementStatus::kUnplaced);
  IdbInstance* createTimingInst(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                                IdbOrient orient = IdbOrient::kN_R0, IdbPlacementStatus status = IdbPlacementStatus::kUnplaced);
  IdbInstance* createUserInst(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                              IdbOrient orient = IdbOrient::kN_R0, IdbPlacementStatus status = IdbPlacementStatus::kUnplaced);
  IdbInstance* insertBufferToNet(string inst_name, string cell_master_name, string net_name, vector<string> pin_name_list);
  IdbInstance* insertCoreFiller(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0);
  IdbInstance* insertIOFiller(string inst_name, string cell_master_name, int32_t coord_x = 0, int32_t coord_y = 0,
                              IdbOrient orient = IdbOrient::kN_R0);

  bool placeInst(string inst_name, int32_t x, int32_t y, string orient, string cell_master_name, string source = "");

  void place_macro_generate_tcl(std::string directory, std::string tcl_name, int number = 100);
  bool place_macro_loc_rand(std::string tcl_path);
  void scale_macro_loc();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iDB design pdn
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iDB design net list
  uint64_t maxFanout();
  uint64_t allNetLength();
  uint64_t netLength(string net_name);
  uint64_t netListLength(vector<IdbNet*>& net_list);
  uint64_t netListLength(vector<string>& net_name_list);
  bool setNetIO(string io_pin_name, string net_name);

  vector<IdbNet*> getSignalNetList();
  vector<IdbNet*> getPdnNetList();
  vector<IdbNet*> getIONetList();

  uint64_t getSignalNetListLength();
  uint64_t getPdnNetListLength();
  uint64_t getIONetListLength();

  IdbPin* getDriverOfNet(IdbNet* net);
  IdbNet* createNet(const string& net_name, IdbConnectType type = IdbConnectType::kNone);
  bool disconnectNet(IdbNet* net);
  bool connectNet(IdbNet* net);

  bool setNetType(string net_name, string type);
  IdbInstance* getIoCellByIoPin(IdbPin* io_pin);

  /// clock net
  vector<IdbNet*> getClockNetList();
  vector<string> getClockNetNameList();
  bool isClockNet(string net_name);
  uint64_t getClockNetListLength();

  /// merge net wire segment
  void mergeNets();
  void mergeNet(IdbNet* net);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Blockage operator
  IdbPlacementBlockage* addPlacementBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury);
  void addPlacementHalo(const string& instance_name, int32_t distance_top, int32_t distance_bottom, int32_t distance_left,
                        int32_t distance_right);
  void removeBlockageExceptPGNet();
  void clearBlockage(string type);

  void addRoutingBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury, const std::vector<std::string>& layers,
                          const bool& is_except_pgnet);
  void addRoutingHalo(const string& instance_name, const std::vector<std::string>& layers, int32_t distance_top, int32_t distance_bottom,
                      int32_t distance_left, int32_t distance_right, const bool& is_except_pgnet);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // pa data
  // bool buildPA(const std::map<std::string, std::map<std::string, std::vector<ids::AccessPoint>>>& master_access_point_map);
  // std::vector<ids::AccessPoint> getMasterPaPointList(std::string master_name, std::string pin_name);
  // std::vector<ids::AccessPoint> getInstancePaPointList(std::string instance_name, std::string pin_name);
  // std::vector<ids::AccessPoint> getInstancePaPointList(std::string cell_master_name, std::string pin_name, int32_t inst_x, int32_t
  // inst_y,
  //                                                      idb::IdbOrient idb_orient);
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// checker
  bool isOnDieBoundary(IdbInstance* io_cell);
  bool isOnDieBoundary(int32_t llx, int32_t lly, int32_t urx, int32_t ury, IdbOrient orient);
  bool isOnIOSite(int32_t llx, int32_t lly, int32_t urx, int32_t ury, IdbOrient orient);
  bool checkInstPlacer(int32_t llx, int32_t lly, int32_t urx, int32_t ury, IdbOrient orient);

  std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> isAllNetConnected();
  bool isNetConnected(std::string net_name);
  bool isNetConnected(IdbNet* net);

 private:
  static DataManager* _instance;
  DataConfig _config;
  IdbBuilder* _idb_builder = nullptr;
  IdbDefService* _idb_def_service = nullptr;
  IdbLefService* _idb_lef_service = nullptr;
  IdbDesign* _design = nullptr;
  IdbLayout* _layout = nullptr;
  // pa
  // std::map<std::string, std::map<std::string, std::vector<ids::AccessPoint>>> _master_access_point_map;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// constructor
  DataManager() {}
  ~DataManager() = default;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// init
  /// iDB init
  bool initConfig(string config_path);
  bool initLef(vector<string> lef_paths, bool b_techlef = false);
  bool initDef(string def_path);
  bool initVerilog(string verilog_path, string top_module);

  /// iDB save
  // bool saveDef(string def_path);

  /// design interface

  /// layout interface

  /// align coordinate
  void transformCoordinate(int32_t& coord_x, int32_t& coord_y, std::string cell_master_name, int32_t inst_x = 0, int32_t inst_y = 0,
                           idb::IdbOrient idb_orient = idb::IdbOrient::kN_R0);
  bool isNeedTransformByDie();
  bool transformByDie();
  bool alignCoord(IdbCoordinate<int32_t>* coord);
  bool alignRect(IdbRect* rect);
  bool alignLayerShape(IdbLayerShape* layer_shape);
  bool alignPin(IdbPin* idb_pin);
  bool alignSignalSegment(IdbRegularWireSegment* idb_segment);
  bool alignSpecialSegment(IdbSpecialWireSegment* idb_segment);
  bool alignVia(IdbVia* idb_via);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// build

  bool wrapPA();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

}  // namespace idm
