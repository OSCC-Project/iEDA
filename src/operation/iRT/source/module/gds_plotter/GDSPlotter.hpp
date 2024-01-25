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
#include "DataManager.hpp"
#include "Database.hpp"
#include "GPBoundary.hpp"
#include "GPGDS.hpp"
#include "GPLYPLayer.hpp"
#include "GPPath.hpp"
#include "GPStruct.hpp"
#include "Stage.hpp"
#include "GPDataType.hpp"

namespace irt {

#define GP_INST (irt::GDSPlotter::getInst())

class GDSPlotter
{
 public:
  static void initInst();
  static GDSPlotter& getInst();
  static void destroyInst();
  // function
  void plot(GPGDS& gp_gds, std::string gds_file_path);
  irt_int getGDSIdxByRouting(irt_int layer_idx);
  irt_int getGDSIdxByCut(irt_int below_layer_idx);

 private:
  // self
  static GDSPlotter* _gp_instance;
  std::map<irt_int, irt_int> _routing_layer_gds_map;
  std::map<irt_int, irt_int> _cut_layer_gds_map;
  std::map<irt_int, irt_int> _gds_routing_layer_map;
  std::map<irt_int, irt_int> _gds_cut_layer_map;

  GDSPlotter() { init(); }
  GDSPlotter(const GDSPlotter& other) = delete;
  GDSPlotter(GDSPlotter&& other) = delete;
  ~GDSPlotter() = default;
  GDSPlotter& operator=(const GDSPlotter& other) = delete;
  GDSPlotter& operator=(GDSPlotter&& other) = delete;
  // function
  void init();
  void buildGDSLayerMap();
  void buildGraphLypFile();
  void writeLypFile(std::string lyp_file_path, std::vector<GPLYPLayer>& lyp_layer_list);
  void buildTopStruct(GPGDS& gp_gds);
  void checkSRefList(GPGDS& gp_gds);
  void plotGDS(GPGDS& gp_gds, std::string gds_file_path);
  void plotStruct(std::ofstream* gds_file, GPStruct& gp_struct);
  void plotBoundary(std::ofstream* gds_file, GPBoundary& gp_boundary);
  void plotPath(std::ofstream* gds_file, GPPath& gp_path);
  void plotText(std::ofstream* gds_file, GPText& gp_text);
  void plotSref(std::ofstream* gds_file, std::string& sref_name);
};
}  // namespace irt
