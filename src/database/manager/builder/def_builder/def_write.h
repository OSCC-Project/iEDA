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
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "../def_service/def_service.h"

namespace idb {

using std::string;
using std::vector;

#define kDbSuccess 0
#define kDbFail 1

#define CLOCKS_PER_MS 1000
enum class DefWriteType
{
  kNone,
  kChip,
  kSynthesis,
  kFloorplan,
  kGlobalPlace,
  kDetailPlace,
  kGlobalRouting,
  kDetailRouting,
  kMax
};

class DefWrite
{
 public:
  DefWrite(IdbDefService* def_service, DefWriteType type = DefWriteType::kChip);
  ~DefWrite();

  // getter
  IdbDefService* get_service() { return _def_service; }

  // writer
  int32_t write_version();
  int32_t write_design();
  int32_t write_units();
  int32_t write_die();
  int32_t write_track_grid();
  int32_t write_row();
  int32_t write_component();
  int32_t write_net();
  int32_t write_net_wire(IdbRegularWire* wire);
  int32_t write_net_wire_segment(IdbRegularWireSegment* segment, string& wire_new_str);
  int32_t write_net_wire_segment_points(IdbRegularWireSegment* segment, string& wire_new_str);
  int32_t write_net_wire_segment_via(IdbRegularWireSegment* segment, string& wire_new_str);
  int32_t write_net_wire_segment_rect(IdbRegularWireSegment* segment, string& wire_new_str);
  int32_t write_special_net();
  int32_t write_specialnet_wire(IdbSpecialWire* wire);
  int32_t write_specialnet_wire_segment(IdbSpecialWireSegment* segment, string& wire_new_str);
  int32_t write_specialnet_wire_segment_points(IdbSpecialWireSegment* segment, string& wire_new_str);
  int32_t write_specialnet_wire_segment_via(IdbSpecialWireSegment* segment, string& wire_new_str);
  int32_t write_specialnet_wire_segment_rect(IdbSpecialWireSegment* segment, string& wire_new_str);
  int32_t write_pin();
  int32_t write_via();
  int32_t write_blockage();
  int32_t write_gcell_grid();
  int32_t write_region();
  int32_t write_slot();
  int32_t write_group();
  int32_t write_fill();
  int32_t write_end();

  void set_start_time(clock_t time) { _start_time = time; }
  void set_end_time(clock_t time) { _end_time = time; }
  float time_eclips() { return (float(_end_time - _start_time)) / CLOCKS_PER_MS; }

  // operator
  bool writeDb(const char* file);
  bool writeChip();
  bool writeDbSynthesis();
  bool initFile(const char* file);
  bool closeFile();

 private:
  IdbDefService* _def_service;
  int32_t _index = 0;
  clock_t _start_time;
  clock_t _end_time;

  FILE* file_write;
  DefWriteType _type;
};
}  // namespace idb
