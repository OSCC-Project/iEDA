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

/*
 * @Author: S.J Chen
 * @Date: 2022-01-21 14:33:51
 * @LastEditTime: 2023-02-22 11:32:34
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/PlacerDB.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_PLACER_BASE_H
#define IPL_PLACER_BASE_H

#include "config/Config.hh"
#include "data/Design.hh"
#include "data/Layout.hh"
#include "module/grid_manager/GridManager.hh"
#include "module/logger/Log.hh"
#include "module/topology_manager/TopologyManager.hh"
#include "module/wrapper/DBWrapper.hh"

namespace ipl {

#define PlacerDBInst (ipl::PlacerDB::getInst())

class PlacerDB
{
 public:
  static PlacerDB& getInst();
  static void destoryInst();
  void initPlacerDB(std::string pl_json_path, DBWrapper* db_wrapper);

  // Place config.
  Config* get_placer_config() { return _config; }

  // Layout.
  const Layout* get_layout() const { return _db_wrapper->get_layout(); }

  // Design.
  Design* get_design() const { return _db_wrapper->get_design(); }

  // Manager.
  TopologyManager* get_topo_manager() const { return _topo_manager; }
  GridManager* get_grid_manager() const { return _grid_manager; }

  // Function.
  void printPlacerDB() const;
  void printLayoutInfo() const;
  void printInstanceInfo() const;
  void printNetInfo() const;
  void printPinInfo() const;
  void printRegionInfo() const;

  void updatePlacerConfig(std::string pl_json_path);
  void updateTopoManager();
  void updateGridManager();
  void updateFromSourceDataBase();
  void updateTopoManager(std::vector<std::string> inst_list);  // TBD.
  void updateGridManager(std::vector<std::string> inst_list);  // TBD.
  void updateFromSourceDataBase(std::vector<std::string> inst_list);
  void updateInstancesForDebug(std::vector<Instance*> inst_list);

  float obtainUtilization();
  void adaptTargetDensity();

  void saveVerilogForDebug(std::string path);

  void writeBackSourceDataBase() { _db_wrapper->writeBackSourceDatabase(); }
  void writeDef(std::string file_name) { _db_wrapper->writeDef(file_name); }

  bool isInitialized() { return _db_wrapper != nullptr; }

  void initTopoManager();
  void initNodes(Design* pl_design);
  void initNetworks(Design* pl_design);
  void initGroups(Design* pl_design);
  void initArcs();
  void generateNetArc(Node* node);
  void generateGroupArc(Node* node);

 public:
  // tmp for iEDA evaluation
  int32_t bin_size_x = 0;
  int32_t bin_size_y = 0;
  int32_t gp_overflow_number = 0;
  float gp_overflow = 0.0f;
  float place_density[3] = {0.0f, 0.0f, 0.0f};
  float pin_density[3] = {0.0f, 0.0f, 0.0f};
  int64_t PL_HPWL[3] = {0, 0, 0};
  int64_t PL_STWL[3] = {0, 0, 0};
  int64_t PL_GRWL[3] = {0, 0, 0};
  int32_t egr_tof[3] = {0,0,0};
  int32_t egr_mof[3] = {0,0,0};
  float egr_ace[3] = {0.0f, 0.0f, 0.0f};
  float tns[3] = {0.0f, 0.0f, 0.0f};
  float wns[3] = {0.0f, 0.0f, 0.0f};
  float suggest_freq[3] = {0.0f, 0.0f, 0.0f};
  int64_t lg_total_movement = 0;
  int64_t lg_max_movement = 0;
  float sta_update_time = 0.0f;
  int32_t init_inst_cnt = 0;

 private:
  static PlacerDB* _s_placer_db_instance;

  Config* _config;
  DBWrapper* _db_wrapper;

  TopologyManager* _topo_manager;
  GridManager* _grid_manager;

  PlacerDB();
  PlacerDB(const PlacerDB&) = delete;
  PlacerDB(PlacerDB&&) = delete;
  ~PlacerDB();
  PlacerDB& operator=(const PlacerDB&) = delete;
  PlacerDB& operator=(PlacerDB&&) = delete;
  void initGridManager();
  void initGridManagerFixedArea();

  void sortDataForParallel();
  void initIgnoreNets(int32_t ignore_net_degree);
};
}  // namespace ipl

#endif