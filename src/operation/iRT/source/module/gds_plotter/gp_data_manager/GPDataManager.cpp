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
#include "GPDataManager.hpp"

#include "GPGraphType.hpp"
#include "GPLYPLayer.hpp"
#include "GPLayoutType.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void GPDataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

// private

void GPDataManager::wrapConfig(Config& config)
{
  _gp_config.temp_directory_path = config.gp_temp_directory_path;
}

void GPDataManager::wrapDatabase(Database& database)
{
  wrapGCellAxis(database);
  wrapDie(database);
  wrapViaLib(database);
  wrapLayerList(database);
  wrapBlockageList(database);
}

void GPDataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _gp_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void GPDataManager::wrapDie(Database& database)
{
  EXTPlanarRect& die = _gp_database.get_die();
  die = database.get_die();
}

void GPDataManager::wrapViaLib(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _gp_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void GPDataManager::wrapLayerList(Database& database)
{
  _gp_database.get_routing_layer_list() = database.get_routing_layer_list();
  _gp_database.get_cut_layer_list() = database.get_cut_layer_list();
}

void GPDataManager::wrapBlockageList(Database& database)
{
  _gp_database.get_routing_blockage_list() = database.get_routing_blockage_list();
  _gp_database.get_cut_blockage_list() = database.get_cut_blockage_list();
}

void GPDataManager::buildConfig()
{
}

void GPDataManager::buildDatabase()
{
  buildGDSLayerMap();
  buildLayoutLypFile();
  buildGraphLypFile();
}

void GPDataManager::buildGDSLayerMap()
{
  std::map<irt_int, irt_int>& routing_layer_gds_map = _gp_database.get_routing_layer_gds_map();
  std::map<irt_int, irt_int>& gds_routing_layer_map = _gp_database.get_gds_routing_layer_map();
  std::map<irt_int, irt_int>& cut_layer_gds_map = _gp_database.get_cut_layer_gds_map();
  std::map<irt_int, irt_int>& gds_cut_layer_map = _gp_database.get_gds_cut_layer_map();

  std::map<irt_int, irt_int> order_gds_map;
  for (RoutingLayer& routing_layer : _gp_database.get_routing_layer_list()) {
    order_gds_map[routing_layer.get_layer_order()] = -1;
  }
  for (CutLayer& cut_layer : _gp_database.get_cut_layer_list()) {
    order_gds_map[cut_layer.get_layer_order()] = -1;
  }
  // 0为die 最后一个为GCell 中间为cut+routing
  irt_int gds_layer_idx = 1;
  for (auto it = order_gds_map.begin(); it != order_gds_map.end(); it++) {
    it->second = gds_layer_idx++;
  }
  for (RoutingLayer& routing_layer : _gp_database.get_routing_layer_list()) {
    irt_int gds_layer_idx = order_gds_map[routing_layer.get_layer_order()];
    routing_layer_gds_map[routing_layer.get_layer_idx()] = gds_layer_idx;
    gds_routing_layer_map[gds_layer_idx] = routing_layer.get_layer_idx();
  }
  for (CutLayer& cut_layer : _gp_database.get_cut_layer_list()) {
    irt_int gds_layer_idx = order_gds_map[cut_layer.get_layer_order()];
    cut_layer_gds_map[cut_layer.get_layer_idx()] = gds_layer_idx;
    gds_cut_layer_map[gds_layer_idx] = cut_layer.get_layer_idx();
  }
}

void GPDataManager::buildLayoutLypFile()
{
  std::vector<RoutingLayer>& routing_layer_list = _gp_database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _gp_database.get_cut_layer_list();
  std::map<irt_int, irt_int>& gds_routing_layer_map = _gp_database.get_gds_routing_layer_map();
  std::map<irt_int, irt_int>& gds_cut_layer_map = _gp_database.get_gds_cut_layer_map();

  std::vector<std::string> color_list = {"#ff9d9d", "#ff80a8", "#c080ff", "#9580ff", "#8086ff", "#80a8ff", "#ff0000", "#ff0080", "#ff00ff",
                                         "#8000ff", "#0000ff", "#0080ff", "#800000", "#800057", "#800080", "#500080", "#000080", "#004080",
                                         "#80fffb", "#80ff8d", "#afff80", "#f3ff80", "#ffc280", "#ffa080", "#00ffff", "#01ff6b", "#91ff00",
                                         "#ddff00", "#ffae00", "#ff8000", "#008080", "#008050", "#008000", "#508000", "#808000", "#805000"};
  std::vector<std::string> pattern_list = {"I5", "I9"};

  std::vector<GPLayoutType> routing_data_type_list
      = {GPLayoutType::kText,      GPLayoutType::kBoundingBox, GPLayoutType::kPort,           GPLayoutType::kAccessPoint,
         GPLayoutType::kGuide,     GPLayoutType::kPreferTrack, GPLayoutType::kNonpreferTrack, GPLayoutType::kWire,
         GPLayoutType::kEnclosure, GPLayoutType::kBlockage,    GPLayoutType::kConnection};
  std::vector<GPLayoutType> cut_data_type_list = {GPLayoutType::kText, GPLayoutType::kCut, GPLayoutType::kBlockage};

  // 0为die 最后一个为GCell 中间为cut+routing
  irt_int gds_layer_size = 2 + static_cast<irt_int>(gds_routing_layer_map.size() + gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (irt_int gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    std::string color = color_list[gds_layer_idx % color_list.size()];
    std::string pattern = pattern_list[gds_layer_idx % pattern_list.size()];

    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(color, pattern, "DIE", gds_layer_idx, 0);
    } else if (gds_layer_idx == (gds_layer_size - 1)) {
      lyp_layer_list.emplace_back(color, pattern, "GCELL", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(color, pattern, "GCELL_text", gds_layer_idx, 1);
    } else {
      if (RTUtil::exist(gds_routing_layer_map, gds_layer_idx)) {
        // routing
        std::string routing_layer_name = routing_layer_list[gds_routing_layer_map[gds_layer_idx]].get_layer_name();
        for (GPLayoutType routing_data_type : routing_data_type_list) {
          lyp_layer_list.emplace_back(color, pattern, RTUtil::getString(routing_layer_name, "_", GetGPLayoutTypeName()(routing_data_type)),
                                      gds_layer_idx, static_cast<irt_int>(routing_data_type));
        }
      } else if (RTUtil::exist(gds_cut_layer_map, gds_layer_idx)) {
        // cut
        std::string cut_layer_name = cut_layer_list[gds_cut_layer_map[gds_layer_idx]].get_layer_name();
        for (GPLayoutType cut_data_type : cut_data_type_list) {
          lyp_layer_list.emplace_back(color, pattern, RTUtil::getString(cut_layer_name, "_", GetGPLayoutTypeName()(cut_data_type)),
                                      gds_layer_idx, static_cast<irt_int>(cut_data_type));
        }
      }
    }
  }
  writeLypFile(_gp_config.temp_directory_path + "layout.lyp", lyp_layer_list);
}

void GPDataManager::buildGraphLypFile()
{
  std::vector<RoutingLayer>& routing_layer_list = _gp_database.get_routing_layer_list();
  std::map<irt_int, irt_int>& gds_routing_layer_map = _gp_database.get_gds_routing_layer_map();
  std::map<irt_int, irt_int>& gds_cut_layer_map = _gp_database.get_gds_cut_layer_map();

  std::vector<std::string> color_list = {"#ff9d9d", "#ff80a8", "#c080ff", "#9580ff", "#8086ff", "#80a8ff", "#ff0000", "#ff0080", "#ff00ff",
                                         "#8000ff", "#0000ff", "#0080ff", "#800000", "#800057", "#800080", "#500080", "#000080", "#004080",
                                         "#80fffb", "#80ff8d", "#afff80", "#f3ff80", "#ffc280", "#ffa080", "#00ffff", "#01ff6b", "#91ff00",
                                         "#ddff00", "#ffae00", "#ff8000", "#008080", "#008050", "#008000", "#508000", "#808000", "#805000"};
  std::vector<std::string> pattern_list = {"I5", "I9"};

  std::vector<GPGraphType> routing_data_type_list
      = {GPGraphType::kNone, GPGraphType::kOpen, GPGraphType::kClose,    GPGraphType::kInfo,       GPGraphType::kNeighbor,
         GPGraphType::kKey,  GPGraphType::kPath, GPGraphType::kBlockage, GPGraphType::kFenceRegion};

  // 0为base_region 最后一个为GCell 中间为cut+routing
  irt_int gds_layer_size = 2 + static_cast<irt_int>(gds_routing_layer_map.size() + gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (irt_int gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    std::string color = color_list[gds_layer_idx % color_list.size()];
    std::string pattern = pattern_list[gds_layer_idx % pattern_list.size()];

    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(color, pattern, "base_region", gds_layer_idx, 0);
    } else if (RTUtil::exist(gds_routing_layer_map, gds_layer_idx)) {
      // routing
      std::string routing_layer_name = routing_layer_list[gds_routing_layer_map[gds_layer_idx]].get_layer_name();
      for (GPGraphType routing_data_type : routing_data_type_list) {
        lyp_layer_list.emplace_back(color, pattern, RTUtil::getString(routing_layer_name, "_", GetGPGraphTypeName()(routing_data_type)),
                                    gds_layer_idx, static_cast<irt_int>(routing_data_type));
      }
    }
  }
  writeLypFile(_gp_config.temp_directory_path + "graph.lyp", lyp_layer_list);
}

void GPDataManager::writeLypFile(std::string lyp_file_path, std::vector<GPLYPLayer>& lyp_layer_list)
{
  std::ofstream* lyp_file = RTUtil::getOutputFileStream(lyp_file_path);
  RTUtil::pushStream(lyp_file, "<?xml version=\"1.0\" encoding=\"utf-8\"?>", "\n");
  RTUtil::pushStream(lyp_file, "<layer-properties>", "\n");

  for (size_t i = 0; i < lyp_layer_list.size(); i++) {
    GPLYPLayer& lyp_layer = lyp_layer_list[i];
    RTUtil::pushStream(lyp_file, "<properties>", "\n");
    RTUtil::pushStream(lyp_file, "<frame-color>", lyp_layer.get_color(), "</frame-color>", "\n");
    RTUtil::pushStream(lyp_file, "<fill-color>", lyp_layer.get_color(), "</fill-color>", "\n");
    RTUtil::pushStream(lyp_file, "<frame-brightness>0</frame-brightness>", "\n");
    RTUtil::pushStream(lyp_file, "<fill-brightness>0</fill-brightness>", "\n");
    RTUtil::pushStream(lyp_file, "<dither-pattern>", lyp_layer.get_pattern(), "</dither-pattern>", "\n");
    RTUtil::pushStream(lyp_file, "<line-style/>", "\n");
    RTUtil::pushStream(lyp_file, "<valid>true</valid>", "\n");
    RTUtil::pushStream(lyp_file, "<visible>true</visible>", "\n");
    RTUtil::pushStream(lyp_file, "<transparent>false</transparent>", "\n");
    RTUtil::pushStream(lyp_file, "<width/>", "\n");
    RTUtil::pushStream(lyp_file, "<marked>false</marked>", "\n");
    RTUtil::pushStream(lyp_file, "<xfill>false</xfill>", "\n");
    RTUtil::pushStream(lyp_file, "<animation>0</animation>", "\n");
    RTUtil::pushStream(lyp_file, "<name>", lyp_layer.get_layer_name(), " ", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(),
                       "</name>", "\n");
    RTUtil::pushStream(lyp_file, "<source>", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "@1</source>", "\n");
    RTUtil::pushStream(lyp_file, "</properties>", "\n");
  }
  RTUtil::pushStream(lyp_file, "</layer-properties>", "\n");
  RTUtil::closeFileStream(lyp_file);
}

}  // namespace irt
