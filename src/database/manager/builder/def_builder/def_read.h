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
 * @file		def_read.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


        There is a def builder to build data structure from def.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "def_service.h"
#include "defiAlias.hpp"
#include "defrReader.hpp"

namespace idb {

using std::string;
using std::vector;

#define kDbSuccess 0
#define kDbFail 1

#define CLOCKS_PER_MS 1000

class DefRead
{
 public:
  DefRead(IdbDefService* def_service);
  ~DefRead();

  // getter
  IdbDefService* get_service() { return _def_service; }
  bool createDb(const char* file);
  bool createDbGzip(const char* gzip_file);
  bool createFloorplanDb(const char* file);

  // callback
  static int32_t versionCallback(defrCallbackType_e type, const char* version, defiUserData data);
  static int32_t designCallback(defrCallbackType_e type, const char* name, defiUserData data);
  static int32_t unitsCallback(defrCallbackType_e type, double d, defiUserData data);
  static int32_t dieAreaCallback(defrCallbackType_e type, defiBox* def_box, defiUserData data);
  static int32_t trackGridCallback(defrCallbackType_e type, defiTrack* def_track, defiUserData data);
  static int32_t rowCallback(defrCallbackType_e type, defiRow* def_row, defiUserData data);
  static int32_t componentsCallback(defrCallbackType_e type, defiComponent* def_component, defiUserData data);
  static int32_t componentNumberCallback(defrCallbackType_e type, int def_num, defiUserData data);
  static int32_t componentEndCallback(defrCallbackType_e type, void*, defiUserData data);
  static int32_t netBeginCallback(defrCallbackType_e type, int def_num, defiUserData data);
  static int32_t netCallback(defrCallbackType_e type, defiNet* def_net, defiUserData data);
  static int32_t netEndCallback(defrCallbackType_e type, void*, defiUserData data);
  static int32_t specialNetBeginCallback(defrCallbackType_e type, int def_num, defiUserData data);
  static int32_t specialNetCallback(defrCallbackType_e type, defiNet* def_net, defiUserData data);
  static int32_t specialNetEndCallback(defrCallbackType_e type, void*, defiUserData data);

  static int32_t pinsBeginCallback(defrCallbackType_e type, int def_num, defiUserData data);
  static int32_t pinCallback(defrCallbackType_e type, defiPin* def_pin, defiUserData data);
  static int32_t pinsEndCallback(defrCallbackType_e type, void*, defiUserData data);

  static int32_t viaBeginCallback(defrCallbackType_e type, int def_num, defiUserData data);
  static int32_t viaCallback(defrCallbackType_e type, defiVia* def_via, defiUserData data);

  static int32_t blockageCallback(defrCallbackType_e type, defiBlockage* def_blockage, defiUserData data);
  static int32_t gcellGridCallback(defrCallbackType_e type, defiGcellGrid* def_grid, defiUserData data);
  static int32_t regionCallback(defrCallbackType_e type, defiRegion* def_region, defiUserData data);
  static int32_t slotsCallback(defrCallbackType_e type, defiSlot* def_slot, defiUserData data);
  static int32_t groupCallback(defrCallbackType_e type, defiGroup* def_group, defiUserData data);

  static int32_t fillsCallback(defrCallbackType_e type, int32_t def_num, defiUserData data);
  static int32_t fillCallback(defrCallbackType_e type, defiFill* def_fill, defiUserData data);

  static int32_t busBitCharsCallBack(defrCallbackType_e c, const char* bus_bit_chars, defiUserData data);

  // parser
  int32_t parse_version(const char* version);
  int32_t parse_design(const char* name);
  int32_t parse_units(double microns);
  int32_t parse_die(defiBox* def_box);
  int32_t parse_track_grid(defiTrack* def_track);
  int32_t parse_row(defiRow* def_row);
  int32_t parse_component_number(int32_t def_component_num);
  int32_t parse_component(defiComponent* def_component);
  int32_t parse_net_number(int32_t def_net_num);
  int32_t parse_net(defiNet* def_net);
  int32_t parse_special_net(defiNet* def_net);
  int32_t parse_pdn(defiNet* def_net);
  int32_t parse_pdn_wire(defiNet* def_net, IdbSpecialWireList* wire_list);
  int32_t parse_pdn_rects(defiNet* def_net, IdbSpecialWireList* wire_list);
  int32_t parse_pin_number(int32_t def_pin_num);
  int32_t parse_pin(defiPin* def_pin);
  int32_t parse_via_num(int32_t via_num);
  int32_t parse_via(defiVia* def_via);
  int32_t parse_blockage(defiBlockage* def_blockage);
  int32_t parse_gcell_grid(defiGcellGrid* def_grid);
  int32_t parse_region(defiRegion* def_region);
  int32_t parse_slot(defiSlot* def_slot);
  int32_t parse_group(defiGroup* def_group);
  int32_t parse_fill_number(int32_t def_fill_num);
  int32_t parse_fill(defiFill* def_fill);
  int32_t parse_bus_bit_chars(const char* bus_bit_chars_str);

  void set_start_time(clock_t time) { _start_time = time; }
  void set_end_time(clock_t time) { _end_time = time; }
  float time_eclips() { return (float(_end_time - _start_time)) / CLOCKS_PER_MS; }

  /// verify
  bool check_type(defrCallbackType_e type);

  /// loger
  void logModule(string mudule, int32_t number = -1)
  {
    logSeperate();
    logNumber(mudule, number);
    logSeperate();
  }
  void logSeperate() { std::cout << "**************************************************************" << std::endl; }
  void logNumber(string mudule, int32_t number = -1)
  {
    std::cout << mudule;
    if (number != -1) {
      std::cout << " number : " << number;
    }
    std::cout << std::endl;
  }
  void logInfo(string info, int32_t number = -1)
  {
    std::cout << info;
    if (number != -1) {
      std::cout << " number : " << number;
    }
    std::cout << std::endl;
  }

 private:
  IdbDefService* _def_service;
  clock_t _start_time;
  clock_t _end_time;

  IdbCellMaster* _cur_cell_master;
};
}  // namespace idb
