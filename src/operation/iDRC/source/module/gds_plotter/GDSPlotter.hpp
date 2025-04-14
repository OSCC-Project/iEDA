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
#include "GPDataType.hpp"
#include "GPGDS.hpp"
#include "GPLYPLayer.hpp"
#include "GPPath.hpp"
#include "GPStruct.hpp"
#include "ViolationType.hpp"

namespace idrc {

#define DRCGP (idrc::GDSPlotter::getInst())

class GDSPlotter
{
 public:
  static void initInst();
  static GDSPlotter& getInst();
  static void destroyInst();
  // function
  void init();
  void plot(GPGDS& gp_gds, std::string gds_file_path);
  int32_t getGDSIdxByRouting(int32_t layer_idx);
  int32_t getGDSIdxByCut(int32_t below_layer_idx);
  GPDataType convertGPDataType(ViolationType violation_type);
  void destroy();

 private:
  // self
  static GDSPlotter* _gp_instance;
  std::map<int32_t, int32_t> _routing_layer_gds_map;
  std::map<int32_t, int32_t> _cut_layer_gds_map;
  std::map<int32_t, int32_t> _gds_routing_layer_map;
  std::map<int32_t, int32_t> _gds_cut_layer_map;

  GDSPlotter() = default;
  GDSPlotter(const GDSPlotter& other) = delete;
  GDSPlotter(GDSPlotter&& other) = delete;
  ~GDSPlotter() = default;
  GDSPlotter& operator=(const GDSPlotter& other) = delete;
  GDSPlotter& operator=(GDSPlotter&& other) = delete;
  // function
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
}  // namespace idrc
