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
#ifndef IDB_BUILDER_LEF_READ
#define IDB_BUILDER_LEF_READ
#pragma once
/**
 * @project		iDB
 * @file		lef_read.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


        There is a lef builder to build data structure from lef.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/adapted/boost_polygon.hpp>
#include <boost/polygon/polygon.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../lef_service/lef_service.h"
#include "lef/lefrReader.hpp"
#include "property_parser/cutlayer_parser.h"
#include "property_parser/property_parser.h"
namespace gtl = boost::polygon;
typedef gtl::point_data<int32_t> GtlPoint;
typedef gtl::rectangle_data<int32_t> GtlRect;
typedef gtl::polygon_with_holes_data<int32_t> PolygonWithHoles;
typedef gtl::polygon_90_data<int32_t> GtlPolygon90;
typedef gtl::polygon_90_set_data<int32_t> GtlPolygon90Set;

namespace idb {

#define kDbSuccess 0
#define kDbFail 1

class LefRead
{
 public:
  LefRead(IdbLefService* lef_service);
  ~LefRead();

  // getter
  IdbLefService* get_service() { return _lef_service; }
  bool createDb(const char* file);

 private:
  string _file_name;
  IdbLefService* _lef_service;
  //!---tbd--------
  IdbCellMaster* _this_cell_master;

  // callback
  static int manufacturingCB(lefrCallbackType_e c, double lef_num, lefiUserData data);
  static int propDefCB(lefrCallbackType_e c, lefiProp* prop, lefiUserData data);
  static int maxStackViaCB(lefrCallbackType_e c, lefiMaxStackVia* maxStack, lefiUserData);
  static int siteCB(lefrCallbackType_e c, lefiSite* lef_site, lefiUserData data);
  static int unitsCB(lefrCallbackType_e c, lefiUnits* lef_unit, lefiUserData data);
  static int layerCB(lefrCallbackType_e c, lefiLayer* lef_layer, lefiUserData data);
  static int macroBeginCB(lefrCallbackType_e c, const char* lef_name, lefiUserData data);
  static int macroCB(lefrCallbackType_e c, lefiMacro* lef_macro, lefiUserData data);
  static int macroEndCB(lefrCallbackType_e c, const char* lef_name, lefiUserData data);
  static int pinCB(lefrCallbackType_e c, lefiPin* lef_pin, lefiUserData data);
  static int obstructionCB(lefrCallbackType_e c, lefiObstruction* lef_obs, lefiUserData data);
  static int viaCB(lefrCallbackType_e c, lefiVia* lef_via, lefiUserData data);
  static int viaRuleCB(lefrCallbackType_e c, lefiViaRule* lef_via_rule, lefiUserData data);
  static int nonDefaultCB(lefrCallbackType_e c, lefiNonDefault* def_nd, lefiUserData data);

  // parser
  int parse_manufacture_grid(double value);
  int parse_property_definition(lefiProp* prop);
  int parse_max_stack_via_lef58(std::string data);
  int parse_max_stack_via(lefiMaxStackVia* maxStack);
  int parse_sites(lefiSite* lef_site);
  int parse_units(lefiUnits* lef_units);
  int parse_layer(lefiLayer* lef_layer);
  int parse_layer_routing(lefiLayer* lef_layer, IdbLayerRouting* layer_routing);
  int parse_layer_cut(lefiLayer* lef_layer, IdbLayerCut* layer_cut);
  int parse_layer_masterslice(lefiLayer* lef_layer, IdbLayerMasterslice* layer_master);
  int parse_layer_overlap(lefiLayer* lef_layer, IdbLayerOverlap* layer_overlap);
  int parse_layer_implant(lefiLayer* lef_layer, IdbLayerImplant* layer_implant);
  int parse_macro_new(const char* macro_name);
  int parse_macro(lefiMacro* lef_macro);
  int parse_macro_reset(const char* name);
  int parse_pin(lefiPin* lef_pin);
  int parse_obs(lefiObstruction* lef_obs);
  int parse_via(lefiVia* lef_via);
  int parse_via_rule(lefiViaRule* lef_via_rule);

  // Tech parse
  int parse_rule();

  // verify
  bool check_type(lefrCallbackType_e type);

  /// operator
  int32_t transAreaDB(double value) { return _lef_service->get_layout()->transAreaDB(value); }
  int32_t transUnitDB(double value) { return _lef_service->get_layout()->transUnitDB(value); }
  std::vector<GtlRect> polygonToRects(lefiGeomPolygon* polygon);
};
}  // namespace idb

#endif  // IDB_BUILDER_LEF_READ
