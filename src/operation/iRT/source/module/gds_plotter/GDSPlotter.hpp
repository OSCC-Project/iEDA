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

#include "Config.hpp"
#include "Database.hpp"
#include "GPBoundary.hpp"
#include "GPDataManager.hpp"
#include "GPGDS.hpp"
#include "GPGraphType.hpp"
#include "GPLayoutType.hpp"
#include "GPPath.hpp"
#include "GPStruct.hpp"
#include "Stage.hpp"

namespace irt {

#define GP_INST (irt::GDSPlotter::getInst())

class GDSPlotter
{
 public:
  static void initInst(Config& config, Database& database);
  static GDSPlotter& getInst();
  static void destroyInst();
  // function
  void plot(Net& net, Stage stage, bool add_layout, bool need_clipping);
  void plot(std::vector<Net>& net_list, Stage stage, bool add_layout, bool need_clipping);
  void plot(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping);
  irt_int getGDSIdxByRouting(irt_int layer_idx);
  irt_int getGDSIdxByCut(irt_int below_layer_idx);

 private:
  // self
  static GDSPlotter* _gp_instance;
  // config & database
  GPDataManager _gp_data_manager;

  GDSPlotter(Config& config, Database& database) { init(config, database); }
  GDSPlotter(const GDSPlotter& other) = delete;
  GDSPlotter(GDSPlotter&& other) = delete;
  ~GDSPlotter() = default;
  GDSPlotter& operator=(const GDSPlotter& other) = delete;
  GDSPlotter& operator=(GDSPlotter&& other) = delete;
  // getter

  // setter

  // function
  void init(Config& config, Database& database);
  void addNetList(GPGDS& gp_gds, std::vector<Net>& net_list, Stage stage);
  void addPinList(GPGDS& gp_gds, GPStruct& net_struct, std::vector<Pin>& pin_list);
  void addPinShapeList(GPStruct& pin_struct, Pin& pin);
  void addAccessPointList(GPStruct& pin_struct, Pin& pin);
  void addBoundingBox(GPGDS& gp_gds, GPStruct& net_struct, BoundingBox& bounding_box);
  void addRTNodeTree(GPGDS& gp_gds, GPStruct& net_struct, MTree<RTNode>& node_tree);
  void addPHYNodeTree(GPGDS& gp_gds, GPStruct& net_struct, MTree<PHYNode>& node_tree);
  void addCostMap(GPGDS& gp_gds, std::vector<Net>& net_list);
  void addCostMap(GPGDS& gp_gds, GPStruct& net_struct, BoundingBox& bounding_box, GridMap<double>& cost_map);
  void plotGDS(GPGDS& gp_gds, std::string gds_file_path, bool add_layout, bool need_clipping);
  PlanarRect getClippingWindow(GPGDS& gp_gds);
  void addLayout(GPGDS& gp_gds, PlanarRect& clipping_window);
  void addDie(GPGDS& gp_gds, PlanarRect& clipping_window);
  void addGCellAxis(GPGDS& gp_gds, PlanarRect& clipping_window);
  void addTrackGrid(GPGDS& gp_gds, PlanarRect& clipping_window);
  void addBlockageList(GPGDS& gp_gds, PlanarRect& clipping_window);
  void buildAndCheckGDS(GPGDS& gp_gds);
  void buildTopStruct(GPGDS& gp_gds);
  void checkSRefList(GPGDS& gp_gds);
  void plotGDS(std::string gds_file_path, GPGDS& gp_gds);
  void plotStruct(std::ofstream* gds_file, GPStruct& gp_struct);
  void plotBoundary(std::ofstream* gds_file, GPBoundary& gp_boundary);
  void plotPath(std::ofstream* gds_file, GPPath& gp_path);
  void plotText(std::ofstream* gds_file, GPText& gp_text);
  void plotSref(std::ofstream* gds_file, std::string& sref_name);
};
}  // namespace irt
