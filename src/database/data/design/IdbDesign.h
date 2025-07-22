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
 * @file		IdbDesign.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe def .
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "IdbLayout.h"
#include "db_design/IdbBlockages.h"
#include "db_design/IdbBus.h"
#include "db_design/IdbBusBitChars.h"
#include "db_design/IdbFill.h"
#include "db_design/IdbGroup.h"
#include "db_design/IdbInstance.h"
#include "db_design/IdbNet.h"
#include "db_design/IdbPins.h"
#include "db_design/IdbRegion.h"
#include "db_design/IdbSlot.h"
#include "db_design/IdbSpecialNet.h"
#include "db_design/IdbTrackGrid.h"
#include "db_design/IdbVias.h"
#include "db_layout/IdbUnits.h"

namespace idb {

class IdbDesign
{
 public:
  IdbDesign(IdbLayout* layout = nullptr);
  ~IdbDesign();

  // getter
  IdbLayout* get_layout() { return _layout; }
  const std::string& get_version() const { return _version; }
  const std::string& get_design_name() const { return _design_name; }
  IdbUnits* get_units() { return _units; }

  IdbInstanceList* get_instance_list() { return _instance_list; }
  IdbPins* get_io_pin_list() { return _io_pin_list; }
  IdbNetList* get_net_list() { return _net_list; }
  IdbVias* get_via_list() { return _via_list; }
  IdbBlockageList* get_blockage_list() { return _blockage_list; }
  IdbRegionList* get_region_list() { return _region_list; }
  IdbSlotList* get_slot_list() { return _slot_list; }
  IdbGroupList* get_group_list() { return _group_list; }
  IdbSpecialNetList* get_special_net_list() { return _special_net_list; }
  IdbFillList* get_fill_list() { return _fill_list; }
  IdbBusBitChars* get_bus_bit_chars() { return _bus_bit_chars; }
  IdbBusList* get_bus_list() { return _bus_list; }

  // setter
  void set_version(std::string version) { _version = version; }
  void set_design_name(std::string name) { _design_name = name; }
  void set_units(IdbUnits* units) { _units = units; }
  void set_instance_list(IdbInstanceList* instance_list) { _instance_list = instance_list; }
  void set_io_pin_list(IdbPins* pin_list) { _io_pin_list = pin_list; }
  void set_net_list(IdbNetList* net_list) { _net_list = net_list; }
  void set_via_list(IdbVias* via_list) { _via_list = via_list; }
  void set_blockage_list(IdbBlockageList* blockage_list) { _blockage_list = blockage_list; }
  void set_region_list(IdbRegionList* region_list) { _region_list = region_list; }
  void set_slot_list(IdbSlotList* slot_list) { _slot_list = slot_list; }
  void set_group_list(IdbGroupList* group_list) { _group_list = group_list; }
  void set_special_net_list(IdbSpecialNetList* net_list) { _special_net_list = net_list; }
  void set_fill_list(IdbFillList* fill_list) { _fill_list = fill_list; }
  void set_bus_bit_chars(IdbBusBitChars* busbit_chars) { _bus_bit_chars = busbit_chars; }

  // operator
  int32_t transUnitDB(double value) { return std::round(_units->get_micron_dbu() * value); }
  double transToUDB(int32_t value) { return ((double) value) / _units->get_micron_dbu(); }

  //   void createDefaultVias(IdbLayers* layers);
  bool connectIOPinToPowerStripe(std::vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer);
  bool connectPowerStripe(std::vector<IdbCoordinate<int32_t>*>& point_list, std::string net_name, std::string layer_name);

 private:
  std::string _version = "5.8";
  std::string _design_name;
  IdbUnits* _units;
  IdbInstanceList* _instance_list;
  IdbPins* _io_pin_list;
  IdbNetList* _net_list;
  IdbVias* _via_list;
  IdbBlockageList* _blockage_list;
  IdbRegionList* _region_list;
  IdbSlotList* _slot_list;
  IdbGroupList* _group_list;
  IdbSpecialNetList* _special_net_list;
  IdbFillList* _fill_list;

  IdbLayout* _layout;
  IdbBusBitChars* _bus_bit_chars = nullptr;
  IdbBusList* _bus_list;
};

}  // namespace idb
