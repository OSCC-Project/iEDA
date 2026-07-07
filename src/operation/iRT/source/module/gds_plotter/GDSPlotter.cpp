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

#include <cmath>
#include <cstdint>
#include <ctime>
#include <initializer_list>
#include <limits>

#include "GPDataType.hpp"
#include "GPLYPLayer.hpp"
#include "MTree.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

namespace irt {

namespace {

constexpr uint8_t kNoData = 0x00;
constexpr uint8_t kBitArray = 0x01;
constexpr uint8_t kInt2 = 0x02;
constexpr uint8_t kInt4 = 0x03;
constexpr uint8_t kReal8 = 0x05;
constexpr uint8_t kAscii = 0x06;
constexpr size_t kGDSFileBufferSize = 16 * 1024 * 1024;

const std::string& getInnovusColor(int32_t layer_idx)
{
  static const std::vector<std::string> color_list = {"#2f3bff", "#e52b2b", "#41e65a", "#f0ee66", "#8e2631",
                                                      "#f2b63f", "#c438d0", "#35d4c8", "#8b5a3c", "#f0e85a"};
  if (layer_idx < 0) {
    layer_idx = 0;
  }
  return color_list[layer_idx % color_list.size()];
}

int32_t getCutUpperRoutingLayerIdx(const std::string& cut_layer_name)
{
  int32_t number = 0;
  bool has_number = false;
  for (char ch : cut_layer_name) {
    if ('0' <= ch && ch <= '9') {
      has_number = true;
      number = number * 10 + (ch - '0');
    }
  }
  return has_number ? number : 0;
}

void appendInt2(std::vector<uint8_t>& data, int32_t value)
{
  uint16_t v = static_cast<uint16_t>(value);
  data.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
  data.push_back(static_cast<uint8_t>(v & 0xff));
}

void appendInt4(std::vector<uint8_t>& data, int32_t value)
{
  uint32_t v = static_cast<uint32_t>(value);
  data.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
  data.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
  data.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
  data.push_back(static_cast<uint8_t>(v & 0xff));
}

void appendReal8(std::vector<uint8_t>& data, double value)
{
  if (value == 0) {
    data.insert(data.end(), 8, 0);
    return;
  }

  uint8_t sign = 0;
  if (value < 0) {
    sign = 0x80;
    value = -value;
  }

  int32_t exponent = 0;
  while (value >= 1.0) {
    value /= 16.0;
    exponent++;
  }
  while (value < 0.0625) {
    value *= 16.0;
    exponent--;
  }

  constexpr uint64_t kMantissaBase = (uint64_t(1) << 56);
  uint64_t mantissa = static_cast<uint64_t>(std::round(value * static_cast<double>(kMantissaBase)));
  if (mantissa == kMantissaBase) {
    mantissa >>= 4;
    exponent++;
  }

  data.push_back(static_cast<uint8_t>(sign | static_cast<uint8_t>(exponent + 64)));
  for (int32_t i = 6; i >= 0; i--) {
    data.push_back(static_cast<uint8_t>((mantissa >> (i * 8)) & 0xff));
  }
}

void writeRecord(std::ofstream* gds_file, uint8_t record_type, uint8_t data_type)
{
  uint8_t header[4];
  header[0] = 0;
  header[1] = 4;
  header[2] = record_type;
  header[3] = data_type;
  gds_file->write(reinterpret_cast<const char*>(header), static_cast<std::streamsize>(sizeof(header)));
}

void writeRecord(std::ofstream* gds_file, uint8_t record_type, uint8_t data_type, const std::vector<uint8_t>& data)
{
  int32_t record_size = 4 + static_cast<int32_t>(data.size());
  if (record_size > std::numeric_limits<uint16_t>::max()) {
    RTLOG.error(Loc::current(), "GDS record is too large!");
  }

  thread_local std::vector<uint8_t> record;
  record.clear();
  record.reserve(record_size);
  appendInt2(record, record_size);
  record.push_back(record_type);
  record.push_back(data_type);
  record.insert(record.end(), data.begin(), data.end());

  gds_file->write(reinterpret_cast<const char*>(record.data()), static_cast<std::streamsize>(record.size()));
}

void writeInt2Record(std::ofstream* gds_file, uint8_t record_type, std::initializer_list<int32_t> value_list)
{
  thread_local std::vector<uint8_t> data;
  data.clear();
  data.reserve(value_list.size() * 2);
  for (int32_t value : value_list) {
    appendInt2(data, value);
  }
  writeRecord(gds_file, record_type, kInt2, data);
}

void writeInt2Record(std::ofstream* gds_file, uint8_t record_type, const std::vector<int32_t>& value_list)
{
  thread_local std::vector<uint8_t> data;
  data.clear();
  data.reserve(value_list.size() * 2);
  for (int32_t value : value_list) {
    appendInt2(data, value);
  }
  writeRecord(gds_file, record_type, kInt2, data);
}

void writeBitArrayRecord(std::ofstream* gds_file, uint8_t record_type, int32_t value)
{
  thread_local std::vector<uint8_t> data;
  data.clear();
  appendInt2(data, value);
  writeRecord(gds_file, record_type, kBitArray, data);
}

void writeInt4Record(std::ofstream* gds_file, uint8_t record_type, std::initializer_list<int32_t> value_list)
{
  thread_local std::vector<uint8_t> data;
  data.clear();
  data.reserve(value_list.size() * 4);
  for (int32_t value : value_list) {
    appendInt4(data, value);
  }
  writeRecord(gds_file, record_type, kInt4, data);
}

void writeReal8Record(std::ofstream* gds_file, uint8_t record_type, std::initializer_list<double> value_list)
{
  thread_local std::vector<uint8_t> data;
  data.clear();
  data.reserve(value_list.size() * 8);
  for (double value : value_list) {
    appendReal8(data, value);
  }
  writeRecord(gds_file, record_type, kReal8, data);
}

void writeStringRecord(std::ofstream* gds_file, uint8_t record_type, const std::string& value)
{
  thread_local std::vector<uint8_t> data;
  data.assign(value.begin(), value.end());
  if ((data.size() % 2) == 1) {
    data.push_back('\0');
  }
  writeRecord(gds_file, record_type, kAscii, data);
}

std::vector<int32_t> getDateList()
{
  std::time_t timestamp = std::time(nullptr);
  std::tm* local_time = std::localtime(&timestamp);
  if (local_time == nullptr) {
    return std::vector<int32_t>(12, 0);
  }

  int32_t year = local_time->tm_year + 1900;
  int32_t month = local_time->tm_mon + 1;
  int32_t day = local_time->tm_mday;
  int32_t hour = local_time->tm_hour;
  int32_t minute = local_time->tm_min;
  int32_t second = local_time->tm_sec;
  return {year, month, day, hour, minute, second, year, month, day, hour, minute, second};
}

}  // namespace

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
    RTLOG.error(Loc::current(), "The instance not initialized!");
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
  if (RTUTIL.exist(_routing_layer_gds_map, routing_layer_idx)) {
    gds_layer_idx = _routing_layer_gds_map[routing_layer_idx];
  } else {
    RTLOG.warn(Loc::current(), "The routing_layer_idx '", routing_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

int32_t GDSPlotter::getGDSIdxByCut(int32_t cut_layer_idx)
{
  int32_t gds_layer_idx = 0;
  if (RTUTIL.exist(_cut_layer_gds_map, cut_layer_idx)) {
    gds_layer_idx = _cut_layer_gds_map[cut_layer_idx];
  } else {
    RTLOG.warn(Loc::current(), "The cut_layer_idx '", cut_layer_idx, "' have not gds_layer_idx!");
  }
  return gds_layer_idx;
}

void GDSPlotter::destroy()
{
}

// private

GDSPlotter* GDSPlotter::_gp_instance = nullptr;

void GDSPlotter::buildGDSLayerMap()
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();

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
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();
  std::string& gp_temp_directory_path = RTDM.getConfig().gp_temp_directory_path;

  std::string base_color = "#b0b0b0";
  std::string routing_pattern = "I5";
  std::string cut_pattern = "I9";

  std::map<GPDataType, bool> routing_data_type_visible_map
      = {{GPDataType::kNone, false},          {GPDataType::kOpen, false},      {GPDataType::kClose, false},
         {GPDataType::kInfo, false},          {GPDataType::kNeighbor, false},  {GPDataType::kShadow, false},
         {GPDataType::kKey, false},           {GPDataType::kGlobalPath, true}, {GPDataType::kDetailedPath, true},
         {GPDataType::kPatch, true},          {GPDataType::kShape, true},      {GPDataType::kAccessPoint, false},
         {GPDataType::kAxis, false},          {GPDataType::kOverflow, false},  {GPDataType::kRouteViolation, false},
         {GPDataType::kPatchViolation, false}};
  std::map<GPDataType, bool> cut_data_type_visible_map = {{GPDataType::kGlobalPath, true}, {GPDataType::kDetailedPath, true}, {GPDataType::kShape, true}};

  // 0为base_region 最后一个为GCell 中间为cut+routing
  int32_t gds_layer_size = 2 + static_cast<int32_t>(_gds_routing_layer_map.size() + _gds_cut_layer_map.size());

  std::vector<GPLYPLayer> lyp_layer_list;
  for (int32_t gds_layer_idx = 0; gds_layer_idx < gds_layer_size; gds_layer_idx++) {
    if (gds_layer_idx == 0) {
      lyp_layer_list.emplace_back(base_color, routing_pattern, true, "base_region", gds_layer_idx, 0);
      lyp_layer_list.emplace_back(base_color, cut_pattern, false, "gcell", gds_layer_idx, 1);
      lyp_layer_list.emplace_back(base_color, routing_pattern, false, "bounding_box", gds_layer_idx, 2);
    } else if (RTUTIL.exist(_gds_routing_layer_map, gds_layer_idx)) {
      // routing
      int32_t routing_layer_idx = _gds_routing_layer_map[gds_layer_idx];
      std::string color = getInnovusColor(routing_layer_idx);
      std::string routing_layer_name = routing_layer_list[routing_layer_idx].get_layer_name();
      for (auto& [routing_data_type, visible] : routing_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, routing_pattern, visible, RTUTIL.getString(routing_layer_name, "_", GetGPDataTypeName()(routing_data_type)),
                                    gds_layer_idx, static_cast<int32_t>(routing_data_type));
      }
    } else if (RTUTIL.exist(_gds_cut_layer_map, gds_layer_idx)) {
      // cut
      int32_t cut_layer_idx = _gds_cut_layer_map[gds_layer_idx];
      std::string cut_layer_name = cut_layer_list[cut_layer_idx].get_layer_name();
      int32_t upper_routing_layer_idx = getCutUpperRoutingLayerIdx(cut_layer_name);
      if (RTUTIL.exist(cut_to_adjacent_routing_map, cut_layer_idx)) {
        for (int32_t routing_layer_idx : cut_to_adjacent_routing_map[cut_layer_idx]) {
          if (upper_routing_layer_idx < routing_layer_idx) {
            upper_routing_layer_idx = routing_layer_idx;
          }
        }
      }
      std::string color = getInnovusColor(upper_routing_layer_idx);
      for (auto& [cut_data_type, visible] : cut_data_type_visible_map) {
        lyp_layer_list.emplace_back(color, cut_pattern, visible, RTUTIL.getString(cut_layer_name, "_", GetGPDataTypeName()(cut_data_type)), gds_layer_idx,
                                    static_cast<int32_t>(cut_data_type));
      }
    }
  }
  writeLypFile(RTUTIL.getString(gp_temp_directory_path, "rt.lyp"), lyp_layer_list);
}

void GDSPlotter::writeLypFile(std::string lyp_file_path, std::vector<GPLYPLayer>& lyp_layer_list)
{
  std::ofstream* lyp_file = RTUTIL.getOutputFileStream(lyp_file_path);
  RTUTIL.pushStream(lyp_file, "<?xml version=\"1.0\" encoding=\"utf-8\"?>", "\n");
  RTUTIL.pushStream(lyp_file, "<layer-properties>", "\n");

  for (size_t i = 0; i < lyp_layer_list.size(); i++) {
    GPLYPLayer& lyp_layer = lyp_layer_list[i];
    RTUTIL.pushStream(lyp_file, "<properties>", "\n");
    RTUTIL.pushStream(lyp_file, "<frame-color>", lyp_layer.get_color(), "</frame-color>", "\n");
    RTUTIL.pushStream(lyp_file, "<fill-color>", lyp_layer.get_color(), "</fill-color>", "\n");
    RTUTIL.pushStream(lyp_file, "<frame-brightness>0</frame-brightness>", "\n");
    RTUTIL.pushStream(lyp_file, "<fill-brightness>0</fill-brightness>", "\n");
    RTUTIL.pushStream(lyp_file, "<dither-pattern>", lyp_layer.get_pattern(), "</dither-pattern>", "\n");
    RTUTIL.pushStream(lyp_file, "<line-style/>", "\n");
    RTUTIL.pushStream(lyp_file, "<valid>true</valid>", "\n");
    if (lyp_layer.get_visible()) {
      RTUTIL.pushStream(lyp_file, "<visible>true</visible>", "\n");
    } else {
      RTUTIL.pushStream(lyp_file, "<visible>false</visible>", "\n");
    }
    RTUTIL.pushStream(lyp_file, "<transparent>false</transparent>", "\n");
    RTUTIL.pushStream(lyp_file, "<width/>", "\n");
    RTUTIL.pushStream(lyp_file, "<marked>false</marked>", "\n");
    RTUTIL.pushStream(lyp_file, "<xfill>false</xfill>", "\n");
    RTUTIL.pushStream(lyp_file, "<animation>0</animation>", "\n");
    RTUTIL.pushStream(lyp_file, "<name>", lyp_layer.get_layer_name(), " ", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "</name>", "\n");
    RTUTIL.pushStream(lyp_file, "<source>", lyp_layer.get_layer_idx(), "/", lyp_layer.get_data_type(), "@1</source>", "\n");
    RTUTIL.pushStream(lyp_file, "</properties>", "\n");
  }
  RTUTIL.pushStream(lyp_file, "</layer-properties>", "\n");
  RTUTIL.closeFileStream(lyp_file);
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
      RTLOG.warn(Loc::current(), "There is no corresponding structure ", nonexistent_sref_name, " in GDS!");
    }
    RTLOG.error(Loc::current(), "There is a non-existent structure reference!");
  }
}

void GDSPlotter::plotGDS(GPGDS& gp_gds, std::string gds_file_path)
{
  Monitor monitor;

  RTLOG.info(Loc::current(), "The gds file is being saved...");

  std::vector<char> file_buffer(kGDSFileBufferSize);
  std::ofstream* gds_file = new std::ofstream();
  gds_file->rdbuf()->pubsetbuf(file_buffer.data(), static_cast<std::streamsize>(file_buffer.size()));
  gds_file->open(gds_file_path, std::ios::out | std::ios::binary);
  if (!gds_file->is_open()) {
    RTLOG.error(Loc::current(), "Failed to open file '", gds_file_path, "'!");
  }

  std::vector<int32_t> date_list = getDateList();
  writeInt2Record(gds_file, 0x00, {600});
  writeInt2Record(gds_file, 0x01, date_list);
  writeStringRecord(gds_file, 0x02, gp_gds.get_top_name());
  writeReal8Record(gds_file, 0x03, {0.001, 1e-9});
  std::vector<GPStruct>& struct_list = gp_gds.get_struct_list();
  for (size_t i = 0; i < struct_list.size(); i++) {
    plotStruct(gds_file, struct_list[i], date_list);
  }
  writeRecord(gds_file, 0x04, kNoData);
  RTUTIL.closeFileStream(gds_file);

  RTLOG.info(Loc::current(), "The gds file has been saved in '", gds_file_path, "'!", monitor.getStatsInfo());
}

void GDSPlotter::plotStruct(std::ofstream* gds_file, GPStruct& gp_struct, const std::vector<int32_t>& date_list)
{
  writeInt2Record(gds_file, 0x05, date_list);
  writeStringRecord(gds_file, 0x06, gp_struct.get_name());
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
  writeRecord(gds_file, 0x07, kNoData);
}

void GDSPlotter::plotBoundary(std::ofstream* gds_file, GPBoundary& gp_boundary)
{
  int32_t ll_x = gp_boundary.get_ll_x();
  int32_t ll_y = gp_boundary.get_ll_y();
  int32_t ur_x = gp_boundary.get_ur_x();
  int32_t ur_y = gp_boundary.get_ur_y();

  writeRecord(gds_file, 0x08, kNoData);
  writeInt2Record(gds_file, 0x0d, {gp_boundary.get_layer_idx()});
  writeInt2Record(gds_file, 0x0e, {gp_boundary.get_data_type()});
  writeInt4Record(gds_file, 0x10, {ll_x, ll_y, ur_x, ll_y, ur_x, ur_y, ll_x, ur_y, ll_x, ll_y});
  writeRecord(gds_file, 0x11, kNoData);
}

void GDSPlotter::plotPath(std::ofstream* gds_file, GPPath& gp_path)
{
  Segment<PlanarCoord>& segment = gp_path.get_segment();
  int32_t first_x = segment.get_first().get_x();
  int32_t first_y = segment.get_first().get_y();
  int32_t second_x = segment.get_second().get_x();
  int32_t second_y = segment.get_second().get_y();

  writeRecord(gds_file, 0x09, kNoData);
  writeInt2Record(gds_file, 0x0d, {gp_path.get_layer_idx()});
  writeInt2Record(gds_file, 0x0e, {gp_path.get_data_type()});
  writeInt4Record(gds_file, 0x0f, {gp_path.get_width()});
  writeInt4Record(gds_file, 0x10, {first_x, first_y, second_x, second_y});
  writeRecord(gds_file, 0x11, kNoData);
}

void GDSPlotter::plotText(std::ofstream* gds_file, GPText& gp_text)
{
  PlanarCoord& coord = gp_text.get_coord();
  int32_t x = coord.get_x();
  int32_t y = coord.get_y();
  int32_t presentation = static_cast<int32_t>(gp_text.get_presentation());

  writeRecord(gds_file, 0x0c, kNoData);
  writeInt2Record(gds_file, 0x0d, {gp_text.get_layer_idx()});
  writeInt2Record(gds_file, 0x16, {gp_text.get_text_type()});
  if (presentation >= 0) {
    writeBitArrayRecord(gds_file, 0x17, presentation);
  }
  writeInt4Record(gds_file, 0x10, {x, y});
  writeStringRecord(gds_file, 0x19, gp_text.get_message());
  writeRecord(gds_file, 0x11, kNoData);
}

void GDSPlotter::plotSref(std::ofstream* gds_file, std::string& sref_name)
{
  writeRecord(gds_file, 0x0a, kNoData);
  writeStringRecord(gds_file, 0x12, sref_name);
  writeInt4Record(gds_file, 0x10, {0, 0});
  writeRecord(gds_file, 0x11, kNoData);
}

}  // namespace irt
