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
 * @project		iDB
 * @file		def_write.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


        There is a def builder to write def file from data structure.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "../def_service/def_service.h"
#include "../lef_service/lef_service.h"
#include "GJWriter.hpp"

namespace idb {

using std::string;
using std::vector;

#define kDbSuccess 0
#define kDbFail 1

#define CLOCKS_PER_MS 1000

class Gds2JsonWrite
{
 public:
  Gds2JsonWrite(IdbDefService* def_service);
  ~Gds2JsonWrite();

  // getter
  IdbDefService* get_service() { return _def_service; }

  // writer
  int32_t set_units();

  int32_t write_version();
  int32_t write_design();
  int32_t write_die();
  int32_t write_track_grid();
  int32_t write_row();
  int32_t write_component();
  int32_t write_net();
  int32_t write_net_wire(JsonStruct* json_struct, IdbRegularWire* wire);
  int32_t write_net_wire_segment(JsonStruct* json_struct, IdbRegularWireSegment* segment);
  int32_t write_net_wire_segment_points(JsonStruct* json_struct, IdbRegularWireSegment* segment);
  int32_t write_net_wire_segment_via(JsonStruct* json_struct, IdbRegularWireSegment* segment);
  int32_t write_net_wire_segment_rect(JsonStruct* json_struct, IdbRegularWireSegment* segment);
  int32_t write_special_net();
  int32_t write_specialnet_wire(JsonStruct* json_struct, IdbSpecialWire* wire);
  int32_t write_specialnet_wire_segment(JsonStruct* json_struct, IdbSpecialWireSegment* segment);
  int32_t write_specialnet_wire_segment_points(JsonStruct* json_struct, IdbSpecialWireSegment* segment);
  int32_t write_specialnet_wire_segment_via(JsonStruct* json_struct, IdbSpecialWireSegment* segment);
  int32_t write_specialnet_wire_segment_rect(JsonStruct* json_struct, IdbSpecialWireSegment* segment);
  int32_t write_pin();
  int32_t write_via();
  int32_t write_blockage();
  int32_t write_gcell_grid();
  int32_t write_region();
  int32_t write_slot();
  int32_t write_group();
  int32_t write_fill();

  void set_start_time(clock_t time) { _start_time = time; }
  void set_end_time(clock_t time) { _end_time = time; }
  float time_eclips() { return (float(_end_time - _start_time)) / CLOCKS_PER_MS; }

  // operator
  bool writeDb(const char* file,string option);
  bool writeChip();
  std::unordered_map<std::string, int> srefHash;

 private:
  IdbDefService* _def_service;
  IdbLefService* _lef_service;
  int32_t _index = 0;
  clock_t _start_time;
  clock_t _end_time;
  FILE* file_write;

  JsonTextWriter _writer;
  JsonData _json;
  JsonStruct* _top_struct;
  void addSRefDefault(string name)
  {
    JsonSref sref;
    sref.add_coord(0, 0);
    sref.sname = name;
    _top_struct->add_element(sref);
  }

  void addStruct(JsonStruct* json_struct);
  void writeStruct();
  void write_diearea();
  bool write_layer();
  void activate_layer(IdbLayers* layer_list);
  int32_t _unit_microns = -1;
  int32_t transDB2Unit(int32_t value)
  {
    return value;
    return value / _unit_microns;
  }

  ///
  void packVia(JsonStruct* json_struct, IdbVia* via);
  void packPin(JsonStruct* json_struct, IdbPin* pin);
  void packLayerShape(JsonStruct* json_struct, IdbLayerShape* layer_shape);
  void packRect(JsonStruct* json_struct, IdbRect* rect, IdbLayer* layer);
  void packRect(JsonStruct* json_struct, IdbRect* rect, int32_t layer_id);
  void packRect(JsonStruct* json_struct, int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, IdbLayer* layer);
  void packSegment(JsonStruct* json_struct, IdbLayerRouting* routing_layer, IdbCoordinate<int32_t>* point_1, IdbCoordinate<int32_t>* point_2,
                   int32_t width = -1);
};


}//name space idb