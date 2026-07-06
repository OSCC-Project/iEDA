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

enum class IdbCreatePolicy
{
  kReturnExisting,
  kErrorIfExists,
  kReplaceExisting
};

struct IdbConnectivityCheckResult
{
  bool ok = true;
  int duplicate_net_count = 0;
  int duplicate_instance_count = 0;
  int duplicate_io_pin_count = 0;
  int stale_regular_pin_ref_count = 0;
  int stale_pin_net_ref_count = 0;
  int stale_special_pin_ref_count = 0;
  int pin_reverse_mismatch_count = 0;
  int net_instance_mismatch_count = 0;
  int duplicate_pin_ref_count = 0;
  int floating_pin_count = 0;
  std::vector<std::string> messages;
};

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
  IdbPin* createOrFindIoPin(const std::string& pin_name, IdbCreatePolicy policy = IdbCreatePolicy::kReturnExisting);
  bool removeIoPinSafe(const std::string& pin_name);
  bool removeIoPinSafe(IdbPin* pin);
  IdbInstance* createInstance(const std::string& inst_name, const std::string& master_name,
                              IdbInstanceType type = IdbInstanceType::kNone,
                              IdbPlacementStatus status = IdbPlacementStatus::kNone,
                              IdbOrient orient = IdbOrient::kNone, int32_t coord_x = 0, int32_t coord_y = 0,
                              IdbCreatePolicy policy = IdbCreatePolicy::kReturnExisting);
  bool placeInstance(const std::string& inst_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                     IdbPlacementStatus status = IdbPlacementStatus::kPlaced);
  bool replaceInstanceMaster(const std::string& inst_name, const std::string& master_name, bool preserve_connection = true);
  bool removeInstanceSafe(const std::string& inst_name);
  std::string makeUniqueInstanceName(const std::string& prefix) const;
  IdbNet* createOrFindNet(const std::string& net_name, IdbConnectType type = IdbConnectType::kNone,
                          IdbCreatePolicy policy = IdbCreatePolicy::kReturnExisting);
  std::string makeUniqueNetName(const std::string& prefix) const;
  bool setNetConnectType(const std::string& net_name, IdbConnectType type);
  bool connectPinToNet(IdbPin* pin, IdbNet* net);
  bool connectIoPinToNet(const std::string& io_pin_name, const std::string& net_name);
  bool connectInstancePinToNet(const std::string& inst_name, const std::string& pin_name, const std::string& net_name);
  bool disconnectPinFromNet(IdbPin* pin);
  bool removeNetSafe(const std::string& net_name);
  bool renameNet(IdbNet* net, const std::string& new_name);
  bool mergeNetInto(const std::string& target_net_name, const std::string& source_net_name, bool move_wires = true);
  IdbSpecialNet* createOrFindSpecialNet(const std::string& net_name, IdbConnectType type = IdbConnectType::kNone,
                                        IdbCreatePolicy policy = IdbCreatePolicy::kReturnExisting);
  bool connectPinToSpecialNet(IdbPin* pin, IdbSpecialNet* net);
  IdbSpecialNet* findSpecialNetForInstancePin(IdbPin* pin) const;
  int32_t connectInstancePinsToSpecialNet(const std::vector<std::string>& pin_name_list, IdbSpecialNet* net);
  int32_t materializeSpecialNetWildcardPins(IdbSpecialNet* net);
  int32_t materializeAllSpecialNetWildcardPins();
  bool disconnectPinFromSpecialNet(IdbPin* pin);
  bool removeSpecialNetSafe(const std::string& net_name);
  void mergeAllNetWires();
  IdbPlacementBlockage* addPlacementBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury);
  void addPlacementHalo(const std::string& instance_name, int32_t distance_top, int32_t distance_bottom, int32_t distance_left,
                        int32_t distance_right);
  void removeBlockageExceptPGNet();
  void clearBlockage(std::string type = "");
  void addRoutingBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury, const std::vector<std::string>& layers,
                          const bool& is_except_pgnet);
  void addRoutingHalo(const std::string& instance_name, const std::vector<std::string>& layers, int32_t distance_top,
                      int32_t distance_bottom, int32_t distance_left, int32_t distance_right, const bool& is_except_pgnet);
  IdbConnectivityCheckResult validateConnectivity(bool check_floating = false) const;
  bool writeConnectivitySnapshot(const std::string& path, bool check_floating = false) const;
  bool connectIOPinToPowerStripe(std::vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer);
  bool connectPowerStripe(std::vector<IdbCoordinate<int32_t>*>& point_list, std::string net_name, std::string layer_name);
  std::vector<std::string> m_instID2Name;
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
