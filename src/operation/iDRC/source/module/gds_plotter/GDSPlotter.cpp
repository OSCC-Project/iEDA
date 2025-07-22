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
#include "GDSPlotter.hpp"

#include "GPDataType.hpp"
#include "GPLYPLayer.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

namespace idrc {

// public

void GDSPlotter::initInst()
{
  if (_gp_instance == nullptr) {
    _gp_instance = new GDSPlotter();
  }
}

GDSPlotter& GDSPlotter::getInst()
{
  if (_gp_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_gp_instance;
}

void GDSPlotter::destroyInst()
{
  if (_gp_instance != nullptr) {
    delete _gp_instance;
    _gp_instance = nullptr;
  }
}

// function

void GDSPlotter::init()
{
  buildGDSLayerMap();
  buildGraphLypFile();
}

void GDSPlotter::plot(GPGDS& gp_gds, std::string gds_file_path)
{
  buildTopStruct(gp_gds);
  checkSRefList(gp_gds);
  plotGDS(gp_gds, gds_file_path);
}

int32_t GDSPlotter::getGDSIdxByRouting(int32_t routing_layer_idx)
{
  int32_t gds_layer_idx = 0;
  if (DRCUTIL.exist(_routing_layer_gds_map, routing_layer_idx)) {
    gds_layer_idx = _routing_layer_gds_map[routing_layer_idx];
  } else {
    DRCLOG.warn(Loc::current(), "The routing_layer_idx '", routing_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

int32_t GDSPlotter::getGDSIdxByCut(int32_t cut_layer_idx)
{
  int32_t gds_layer_idx = 0;
  if (DRCUTIL.exist(_cut_layer_gds_map, cut_layer_idx)) {
    gds_layer_idx = _cut_layer_gds_map[cut_layer_idx];
  } else {
    DRCLOG.warn(Loc::current(), "The cut_layer_idx '", cut_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

GPDataType GDSPlotter::convertGPDataType(ViolationType violation_type)
{
  GPDataType gp_data_type;
  switch (violation_type) {
    case ViolationType::kAdjacentCutSpacing:
      gp_data_type = GPDataType::kAdjacentCutSpacing;
      break;
    case ViolationType::kCornerFillSpacing:
      gp_data_type = GPDataType::kCornerFillSpacing;
      break;
    case ViolationType::kCornerSpacing:
      gp_data_type = GPDataType::kCornerSpacing;
      break;
    case ViolationType::kCutEOLSpacing:
      gp_data_type = GPDataType::kCutEOLSpacing;
      break;
    case ViolationType::kCutShort:
      gp_data_type = GPDataType::kCutShort;
      break;
    case ViolationType::kDifferentLayerCutSpacing:
      gp_data_type = GPDataType::kDifferentLayerCutSpacing;
      break;
    case ViolationType::kEndOfLineSpacing:
      gp_data_type = GPDataType::kEndOfLineSpacing;
      break;
    case ViolationType::kEnclosure:
      gp_data_type = GPDataType::kEnclosure;
      break;
    case ViolationType::kEnclosureEdge:
      gp_data_type = GPDataType::kEnclosureEdge;
      break;
    case ViolationType::kEnclosureParallel:
      gp_data_type = GPDataType::kEnclosureParallel;
      break;
    case ViolationType::kFloatingPatch:
      gp_data_type = GPDataType::kFloatingPatch;
      break;
    case ViolationType::kJogToJogSpacing:
      gp_data_type = GPDataType::kJogToJogSpacing;
      break;
    case ViolationType::kMaximumWidth:
      gp_data_type = GPDataType::kMaximumWidth;
      break;
    case ViolationType::kMaxViaStack:
      gp_data_type = GPDataType::kMaxViaStack;
      break;
    case ViolationType::kMetalShort:
      gp_data_type = GPDataType::kMetalShort;
      break;
    case ViolationType::kMinHole:
      gp_data_type = GPDataType::kMinHole;
      break;
    case ViolationType::kMinimumArea:
      gp_data_type = GPDataType::kMinimumArea;
      break;
    case ViolationType::kMinimumCut:
      gp_data_type = GPDataType::kMinimumCut;
      break;
    case ViolationType::kMinimumWidth:
      gp_data_type = GPDataType::kMinimumWidth;
      break;
    case ViolationType::kMinStep:
      gp_data_type = GPDataType::kMinStep;
      break;
    case ViolationType::kNonsufficientMetalOverlap:
      gp_data_type = GPDataType::kNonsufficientMetalOverlap;
      break;
    case ViolationType::kNotchSpacing:
      gp_data_type = GPDataType::kNotchSpacing;
      break;
    case ViolationType::kOffGridOrWrongWay:
      gp_data_type = GPDataType::kOffGridOrWrongWay;
      break;
    case ViolationType::kOutOfDie:
      gp_data_type = GPDataType::kOutOfDie;
      break;
    case ViolationType::kParallelRunLengthSpacing:
      gp_data_type = GPDataType::kParallelRunLengthSpacing;
      break;
    case ViolationType::kSameLayerCutSpacing:
      gp_data_type = GPDataType::kSameLayerCutSpacing;
      break;
    default:
      DRCLOG.error(Loc::current(), "The violation_type not support!");
      break;
  }
  return gp_data_type;
}

void GDSPlotter::destroy()
{
}

// private

GDSPlotter* GDSPlotter::_gp_instance = nullptr;

void GDSPlotter::buildGDSLayerMap()
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  std::map<int32_t, int32_t> order_gds_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    order_gds_map[routing_layer.get_layer_order()] = -1;
  }
  for (CutLayer& cut_layer : cut_layer_list) {
    order_gds_map[cut_layer.get_layer_order()] = -1;
  }
  // 0为die 最后一个为GCell 中间为cut+routing
  int32_t gds_layer_idx = 1;
  for (auto it = order_gds_map.begin(); it != order_gds_map.end(); it++) {
    it->second = gds_layer_idx++;
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t gds_layer_idx = order_gds_map[routing_layer.get_layer_order()];
    _routing_layer_gds_map[routing_layer.get_layer_idx()] = gds_layer_idx;
    _gds_routing_layer_map[gds_layer_idx] = routing_layer.get_layer_idx();
  }
  for (CutLayer& cut_layer : cut_layer_list) {
    int32_t gds_layer_idx = order_gds_map[cut_layer.get_layer_order()];
    _cut_layer_gds_map[cut_layer.get_layer_idx()] = gds_layer_idx;
    _gds_cut_layer_map[gds_layer_idx] = cut_layer.get_layer_idx();
  }
}

void GDSPlotter::buildGraphLypFile()
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::string& gp_temp_directory_path = DRCDM.getConfig().gp_temp_directory_path;

  std::vector<std::string> color_list
      = {"#ff9d9d", "#ff80a8", "#c080ff", "#9580ff", "#8086ff", "#80a8ff", "#ff0000", "#ff0080", "#ff00ff", "#8000ff", "#0000ff", "#0080ff",
         "#800000", "#800057", "#800080", "#500080", "#000080", "#004080", "#80fffb", "#80ff8d", "#afff80", "#f3ff80", "#ffc280", "#ffa080",
         "#00ffff", "#01ff6b", "#91ff00", "#ddff00", "#ffae00", "#ff8000", "#008080", "#008050", "#008000", "#508000", "#808000", "#805000"};
  std::vector<std::string> pattern_list = {"I5", "I9"};

  std::map<GPDataType, bool> routing_data_type_visible_map = {{GPDataType::kEnvShape, false},
                                                              {GPDataType::kResultShape, false},
                                                              {GPDataType::kAdjacentCutSpacing, false},
                                                              {GPDataType::kCornerFillSpacing, false},
                                                              {GPDataType::kCornerSpacing, false},
                                                              {GPDataType::kCutEOLSpacing, false},
                                                              {GPDataType::kCutShort, false},
                                                              {GPDataType::kDifferentLayerCutSpacing, false},
                                                              {GPDataType::kEndOfLineSpacing, false},
                                                              {GPDataType::kEnclosure, false},
                                                              {GPDataType::kEnclosureEdge, false},
                                                              {GPDataType::kEnclosureParallel, false},
                                                              {GPDataType::kFloatingPatch, false},
                                                              {GPDataType::kJogToJogSpacing, false},
                                                              {GPDataType::kMaximumWidth, false},
                                                              {GPDataType::kMaxViaStack, false},
                                                              {GPDataType::kMetalShort, false},
                                                              {GPDataType::kMinHole, false},
                                                              {GPDataType::kMinimumArea, false},
                                                              {GPDataType::kMinimumCut, false},
                                                              {GPDataType::kMinimumWidth, false},
                                                              {GPDataType::kMinStep, false},
                                                              {GPDataType::kNonsufficientMetalOverlap, false},
                                                              {GPDataType::kNotchSpacing, false},
                                                              {GPDataType::kOffGridOrWrongWay, false},
                                                              {GPDataType::kOutOfDie, false},
                                                              {GPDataType::kParallelRunLengthSpacing, false},
                                                              {GPDataType::kSameLayerCutSpacing, false}};
  std::map<GPDataType, bool> cut_data_type_visible_map = {{GPDataType::kEnvShape, false}, {GPDataType::kResultShape, false}};

  // 0为base_region 最后一个为GCell 中间为cut+routing
  int32_t gds_layer_size = 2 + static_cast<int32_t>(_gds_routing_layer_map.size() + _gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (int32_t gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    std::string color = color_list[gds_layer_idx % color_list.size()];
    std::string pattern = pattern_list[gds_layer_idx % pattern_list.size()];

    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(color, pattern, true, "base_region", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(color, pattern, false, "gcell", gds_layer_idx, 1);
      lyp_layer_list.emplace_back(color, pattern, false, "bounding_box", gds_layer_idx, 2);
    } else if (DRCUTIL.exist(_gds_routing_layer_map, gds_layer_idx)) {
      // routing
      std::string routing_layer_name = routing_layer_list[_gds_routing_layer_map[gds_layer_idx]].get_layer_name();
      for (auto& [routing_data_type, visible] : routing_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, pattern, visible, DRCUTIL.getString(routing_layer_name, "_", GetGPDataTypeName()(routing_data_type)), gds_layer_idx,
                                    static_cast<int32_t>(routing_data_type));
      }
    } else if (DRCUTIL.exist(_gds_cut_layer_map, gds_layer_idx)) {
      // cut
      std::string cut_layer_name = cut_layer_list[_gds_cut_layer_map[gds_layer_idx]].get_layer_name();
      for (auto& [cut_data_type, visible] : cut_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, pattern, visible, DRCUTIL.getString(cut_layer_name, "_", GetGPDataTypeName()(cut_data_type)), gds_layer_idx,
                                    static_cast<int32_t>(cut_data_type));
      }
    }
  }
  writeLypFile(DRCUTIL.getString(gp_temp_directory_path, "drc.lyp"), lyp_layer_list);
}

void GDSPlotter::writeLypFile(std::string lyp_file_path, std::vector<GPLYPLayer>& lyp_layer_list)
{
  std::ofstream* lyp_file = DRCUTIL.getOutputFileStream(lyp_file_path);
  DRCUTIL.pushStream(lyp_file, "<?xml version=\"1.0\" encoding=\"utf-8\"?>", "\n");
  DRCUTIL.pushStream(lyp_file, "<layer-properties>", "\n");

  for (size_t i = 0; i < lyp_layer_list.size(); i++) {
    GPLYPLayer& lyp_layer = lyp_layer_list[i];
    DRCUTIL.pushStream(lyp_file, "<properties>", "\n");
    DRCUTIL.pushStream(lyp_file, "<frame-color>", lyp_layer.get_color(), "</frame-color>", "\n");
    DRCUTIL.pushStream(lyp_file, "<fill-color>", lyp_layer.get_color(), "</fill-color>", "\n");
    DRCUTIL.pushStream(lyp_file, "<frame-brightness>0</frame-brightness>", "\n");
    DRCUTIL.pushStream(lyp_file, "<fill-brightness>0</fill-brightness>", "\n");
    DRCUTIL.pushStream(lyp_file, "<dither-pattern>", lyp_layer.get_pattern(), "</dither-pattern>", "\n");
    DRCUTIL.pushStream(lyp_file, "<line-style/>", "\n");
    DRCUTIL.pushStream(lyp_file, "<valid>true</valid>", "\n");
    if (lyp_layer.get_visible()) {
      DRCUTIL.pushStream(lyp_file, "<visible>true</visible>", "\n");
    } else {
      DRCUTIL.pushStream(lyp_file, "<visible>false</visible>", "\n");
    }
    DRCUTIL.pushStream(lyp_file, "<transparent>false</transparent>", "\n");
    DRCUTIL.pushStream(lyp_file, "<width/>", "\n");
    DRCUTIL.pushStream(lyp_file, "<marked>false</marked>", "\n");
    DRCUTIL.pushStream(lyp_file, "<xfill>false</xfill>", "\n");
    DRCUTIL.pushStream(lyp_file, "<animation>0</animation>", "\n");
    DRCUTIL.pushStream(lyp_file, "<name>", lyp_layer.get_layer_name(), " ", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "</name>", "\n");
    DRCUTIL.pushStream(lyp_file, "<source>", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "@1</source>", "\n");
    DRCUTIL.pushStream(lyp_file, "</properties>", "\n");
  }
  DRCUTIL.pushStream(lyp_file, "</layer-properties>", "\n");
  DRCUTIL.closeFileStream(lyp_file);
}

void GDSPlotter::buildTopStruct(GPGDS& gp_gds)
{
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();

  std::set<std::string> no_ref_struct_name_set;
  for (GPStruct& gp_struct : struct_list) {
    no_ref_struct_name_set.insert(gp_struct.get_name());
  }

  for (GPStruct& gp_struct : struct_list) {
    std::vector<std::string>& sref_name_list = gp_struct.get_sref_name_list();
    for (std::string& sref_name : sref_name_list) {
      no_ref_struct_name_set.erase(sref_name);
    }
  }

  GPStruct top_struct(gp_gds.get_top_name());
  for (const std::string& no_ref_struct_name : no_ref_struct_name_set) {
    top_struct.push(no_ref_struct_name);
  }
  gp_gds.addStruct(top_struct);
}

void GDSPlotter::checkSRefList(GPGDS& gp_gds)
{
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();

  std::set<std::string> nonexistent_sref_name_set;
  for (GPStruct& gp_struct : struct_list) {
    for (std::string& sref_name : gp_struct.get_sref_name_list()) {
      nonexistent_sref_name_set.insert(sref_name);
    }
  }
  for (GPStruct& gp_struct : struct_list) {
    nonexistent_sref_name_set.erase(gp_struct.get_name());
  }

  if (!nonexistent_sref_name_set.empty()) {
    for (const std::string& nonexistent_sref_name : nonexistent_sref_name_set) {
      DRCLOG.warn(Loc::current(), "There is no corresponding structure ", nonexistent_sref_name, " in GDS!");
    }
    DRCLOG.error(Loc::current(), "There is a non-existent structure reference!");
  }
}

void GDSPlotter::plotGDS(GPGDS& gp_gds, std::string gds_file_path)
{
  Monitor monitor;

  DRCLOG.info(Loc::current(), "The gds file is being saved...");

  std::ofstream* gds_file = DRCUTIL.getOutputFileStream(gds_file_path);
  DRCUTIL.pushStream(gds_file, "HEADER 600", "\n");
  DRCUTIL.pushStream(gds_file, "BGNLIB", "\n");
  DRCUTIL.pushStream(gds_file, "LIBNAME ", gp_gds.get_top_name(), "\n");
  DRCUTIL.pushStream(gds_file, "UNITS 0.001 1e-9", "\n");
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();
  for (size_t i = 0; i < struct_list.size(); i++) {
    plotStruct(gds_file, struct_list[i]);
  }
  DRCUTIL.pushStream(gds_file, "ENDLIB", "\n");
  DRCUTIL.closeFileStream(gds_file);

  DRCLOG.info(Loc::current(), "The gds file has been saved in '", gds_file_path, "'!", monitor.getStatsInfo());
}

void GDSPlotter::plotStruct(std::ofstream* gds_file, GPStruct& gp_struct)
{
  DRCUTIL.pushStream(gds_file, "BGNSTR", "\n");
  DRCUTIL.pushStream(gds_file, "STRNAME ", gp_struct.get_name(), "\n");
  // boundary
  for (GPBoundary& gp_boundary : gp_struct.get_boundary_list()) {
    plotBoundary(gds_file, gp_boundary);
  }
  // path
  for (GPPath& gp_path : gp_struct.get_path_list()) {
    plotPath(gds_file, gp_path);
  }
  // text
  for (GPText& gp_text : gp_struct.get_text_list()) {
    plotText(gds_file, gp_text);
  }
  // sref
  for (std::string& sref_name : gp_struct.get_sref_name_list()) {
    plotSref(gds_file, sref_name);
  }
  DRCUTIL.pushStream(gds_file, "ENDSTR", "\n");
}

void GDSPlotter::plotBoundary(std::ofstream* gds_file, GPBoundary& gp_boundary)
{
  int32_t ll_x = gp_boundary.get_ll_x();
  int32_t ll_y = gp_boundary.get_ll_y();
  int32_t ur_x = gp_boundary.get_ur_x();
  int32_t ur_y = gp_boundary.get_ur_y();

  DRCUTIL.pushStream(gds_file, "BOUNDARY", "\n");
  DRCUTIL.pushStream(gds_file, "LAYER ", gp_boundary.get_layer_idx(), "\n");
  DRCUTIL.pushStream(gds_file, "DATATYPE ", static_cast<int32_t>(gp_boundary.get_data_type()), "\n");
  DRCUTIL.pushStream(gds_file, "XY", "\n");
  DRCUTIL.pushStream(gds_file, ll_x, " : ", ll_y, "\n");
  DRCUTIL.pushStream(gds_file, ur_x, " : ", ll_y, "\n");
  DRCUTIL.pushStream(gds_file, ur_x, " : ", ur_y, "\n");
  DRCUTIL.pushStream(gds_file, ll_x, " : ", ur_y, "\n");
  DRCUTIL.pushStream(gds_file, ll_x, " : ", ll_y, "\n");
  DRCUTIL.pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotPath(std::ofstream* gds_file, GPPath& gp_path)
{
  Segment<PlanarCoord>& segment = gp_path.get_segment();
  int32_t first_x = segment.get_first().get_x();
  int32_t first_y = segment.get_first().get_y();
  int32_t second_x = segment.get_second().get_x();
  int32_t second_y = segment.get_second().get_y();

  DRCUTIL.pushStream(gds_file, "PATH", "\n");
  DRCUTIL.pushStream(gds_file, "LAYER ", gp_path.get_layer_idx(), "\n");
  DRCUTIL.pushStream(gds_file, "DATATYPE ", static_cast<int32_t>(gp_path.get_data_type()), "\n");
  DRCUTIL.pushStream(gds_file, "WIDTH ", gp_path.get_width(), "\n");
  DRCUTIL.pushStream(gds_file, "XY", "\n");
  DRCUTIL.pushStream(gds_file, first_x, " : ", first_y, "\n");
  DRCUTIL.pushStream(gds_file, second_x, " : ", second_y, "\n");
  DRCUTIL.pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotText(std::ofstream* gds_file, GPText& gp_text)
{
  PlanarCoord& coord = gp_text.get_coord();
  int32_t x = coord.get_x();
  int32_t y = coord.get_y();

  DRCUTIL.pushStream(gds_file, "TEXT", "\n");
  DRCUTIL.pushStream(gds_file, "LAYER ", gp_text.get_layer_idx(), "\n");
  DRCUTIL.pushStream(gds_file, "TEXTTYPE ", gp_text.get_text_type(), "\n");
  DRCUTIL.pushStream(gds_file, "PRESENTATION ", static_cast<int32_t>(gp_text.get_presentation()), "\n");
  DRCUTIL.pushStream(gds_file, "XY", "\n");
  DRCUTIL.pushStream(gds_file, x, " : ", y, "\n");
  DRCUTIL.pushStream(gds_file, "STRING ", gp_text.get_message(), "\n");
  DRCUTIL.pushStream(gds_file, "ENDEL", "\n");
}

void GDSPlotter::plotSref(std::ofstream* gds_file, std::string& sref_name)
{
  DRCUTIL.pushStream(gds_file, "SREF", "\n");
  DRCUTIL.pushStream(gds_file, "SNAME ", sref_name, "\n");
  DRCUTIL.pushStream(gds_file, "XY 0:0", "\n");
  DRCUTIL.pushStream(gds_file, "ENDEL", "\n");
}

}  // namespace idrc
