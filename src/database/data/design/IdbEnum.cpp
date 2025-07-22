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
/**
 * @project		iDB
 * @file		IdbEnum.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe all idb enum information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbEnum.h"

#include <algorithm>

using namespace std;

namespace idb {

IdbEnum* IdbEnum::_instance = nullptr;
std::mutex IdbEnum::_mutex;

IdbEnum::IdbEnum()
{
  _property_map = new IdbInstancePropertyMap();
  _site_property = new IdbSiteProperty();
  _term_property = new IdbConnectProperty();
  _layer_property = new IdbLayerProperty();
  _cell_property = new IdbCellProperty();
  _region_property = new IdbRegionProperty();
}

IdbEnum::~IdbEnum()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbInstancePropertyMap::IdbInstancePropertyMap()
{
  _type_list = {{IdbInstanceType::kNone, "NONE"}, {IdbInstanceType::kNetlist, "NETLIST"}, {IdbInstanceType::kDist, "DIST"},
                {IdbInstanceType::kUser, "USER"}, {IdbInstanceType::kTiming, "TIMING"},   {IdbInstanceType::kTest, "TEST"}};

  _status_list = {{IdbPlacementStatus::kNone, "NONE"},
                  {IdbPlacementStatus::kFixed, "FIXED"},
                  {IdbPlacementStatus::kCover, "COVER"},
                  {IdbPlacementStatus::kPlaced, "PLACED"},
                  {IdbPlacementStatus::kUnplaced, "UNPLACED"}};

  _status_enum_list = {{IdbPlacementStatus::kNone, 0},
                       {IdbPlacementStatus::kFixed, DEFI_COMPONENT_FIXED},
                       {IdbPlacementStatus::kCover, DEFI_COMPONENT_COVER},
                       {IdbPlacementStatus::kPlaced, DEFI_COMPONENT_PLACED},
                       {IdbPlacementStatus::kUnplaced, DEFI_COMPONENT_UNPLACED}};
}

IdbInstanceType IdbInstancePropertyMap::get_type(string type_name)
{
  auto result = std::find_if(_type_list.begin(), _type_list.end(), [type_name](const auto& iter) { return iter.second == type_name; });

  if (result == _type_list.end()) {
    return IdbInstanceType::kNone;
  }

  return result->first;
}

string IdbInstancePropertyMap::get_type_str(IdbInstanceType type)
{
  auto iter = _type_list.find(type);
  if (iter == _type_list.end()) {
    return string("");
  }

  return iter->second;
}

IdbPlacementStatus IdbInstancePropertyMap::get_status(string status_name)
{
  auto result
      = std::find_if(_status_list.begin(), _status_list.end(), [status_name](const auto& iter) { return iter.second == status_name; });

  if (result == _status_list.end()) {
    return IdbPlacementStatus::kNone;
  }

  return result->first;
}

string IdbInstancePropertyMap::get_status_str(IdbPlacementStatus status)
{
  auto iter = _status_list.find(status);
  if (iter == _status_list.end()) {
    return string("");
  }

  return iter->second;
}

IdbPlacementStatus IdbInstancePropertyMap::get_status(int32_t status_def_enum)
{
  auto result = std::find_if(_status_enum_list.begin(), _status_enum_list.end(),
                             [status_def_enum](const auto& iter) { return iter.second == status_def_enum; });

  if (result == _status_enum_list.end()) {
    return IdbPlacementStatus::kNone;
  }

  return result->first;
}

int32_t IdbInstancePropertyMap::get_status_enum(IdbPlacementStatus status)
{
  auto iter = _status_enum_list.find(status);
  if (iter == _status_enum_list.end()) {
    return 0;
  }

  return iter->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbSiteProperty::IdbSiteProperty()
{
  _site_class_list = {{IdbSiteClass::kCore, "CORE"}, {IdbSiteClass::kPad, "PAD"}};

  _orient_list = {{DEF_ORIENT_N, IdbOrient::kN_R0},   {DEF_ORIENT_W, IdbOrient::kW_R90},   {DEF_ORIENT_S, IdbOrient::kS_R180},
                  {DEF_ORIENT_E, IdbOrient::kE_R270}, {DEF_ORIENT_FN, IdbOrient::kFN_MY},  {DEF_ORIENT_FW, IdbOrient::kFW_MX90},
                  {DEF_ORIENT_FS, IdbOrient::kFS_MX}, {DEF_ORIENT_FE, IdbOrient::kFE_MY90}};

  _orient_string_list = {{IdbOrient::kN_R0, "N"},   {IdbOrient::kW_R90, "W"},    {IdbOrient::kS_R180, "S"}, {IdbOrient::kE_R270, "E"},
                         {IdbOrient::kFN_MY, "FN"}, {IdbOrient::kFW_MX90, "FW"}, {IdbOrient::kFS_MX, "FS"}, {IdbOrient::kFE_MY90, "FE"}};

  _orient_alias_list
      = {{IdbOrient::kN_R0, "R0"},  {IdbOrient::kW_R90, "R90"},    {IdbOrient::kS_R180, "R180"}, {IdbOrient::kE_R270, "R270"},
         {IdbOrient::kFN_MY, "MY"}, {IdbOrient::kFW_MX90, "MX90"}, {IdbOrient::kFS_MX, "MX"},    {IdbOrient::kFE_MY90, "MY90"}};
}

IdbSiteClass IdbSiteProperty::get_class_type(string class_name)
{
  std::transform(class_name.begin(), class_name.end(), class_name.begin(), ::toupper);
  auto result = std::find_if(_site_class_list.begin(), _site_class_list.end(),
                             [class_name](const auto& iter) { return iter.second == class_name; });

  if (result == _site_class_list.end()) {
    return IdbSiteClass::kNone;
  }

  return result->first;
}

string IdbSiteProperty::get_class_name(IdbSiteClass class_type)
{
  auto iter = _site_class_list.find(class_type);
  if (iter == _site_class_list.end()) {
    return string("");
  }

  return iter->second;
}

int32_t IdbSiteProperty::get_orient_def_value(IdbOrient idb_value)
{
  auto result = std::find_if(_orient_list.begin(), _orient_list.end(), [idb_value](const auto& iter) { return iter.second == idb_value; });

  if (result == _orient_list.end()) {
    return -1;
  }

  return result->first;
}

IdbOrient IdbSiteProperty::get_orient_idb_value(int32_t def_value)
{
  auto iter = _orient_list.find(def_value);
  if (iter == _orient_list.end()) {
    return IdbOrient::kNone;
  }

  return iter->second;
}

IdbOrient IdbSiteProperty::get_orient_value(string orient_name)
{
  auto result = std::find_if(_orient_string_list.begin(), _orient_string_list.end(),
                             [orient_name](const auto& iter) { return iter.second == orient_name; });
  if (result != _orient_string_list.end()) {
    return result->first;
  }

  auto result_alias = std::find_if(_orient_alias_list.begin(), _orient_alias_list.end(),
                                   [orient_name](const auto& iter) { return iter.second == orient_name; });
  if (result_alias != _orient_alias_list.end()) {
    return result_alias->first;
  }

  return IdbOrient::kNone;
}

string IdbSiteProperty::get_orient_name(IdbOrient oreint_value)
{
  auto iter = _orient_string_list.find(oreint_value);
  if (iter == _orient_string_list.end()) {
    return string("");
  }

  return iter->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbConnectProperty::IdbConnectProperty()
{
  _direction_list = {{IdbConnectDirection::kInput, "INPUT"},
                     {IdbConnectDirection::kOutput, "OUTPUT"},
                     {IdbConnectDirection::kOutputTriState, "OUTPUT TRISTATE"},
                     {IdbConnectDirection::kInOut, "INOUT"},
                     {IdbConnectDirection::kFeedThru, "FEEDTHRU"}};

  _type_list = {{IdbConnectType::kSignal, "SIGNAL"}, {IdbConnectType::kAnalog, "ANALOG"}, {IdbConnectType::kPower, "POWER"},
                {IdbConnectType::kGround, "GROUND"}, {IdbConnectType::kClock, "CLOCK"},   {IdbConnectType::kTieOff, "TIEOFF"},
                {IdbConnectType::kScan, "SCAN"},     {IdbConnectType::kReset, "RESET"}};

  _pin_shape_list = {{IdbTermShape::kAbutment, "ABUTMENT"}, {IdbTermShape::kRing, "RING"}, {IdbTermShape::kFeedThru, "FEEDTHRU"}};

  _port_class_list = {{IdbPortClass::kNone, "NONE"}, {IdbPortClass::kCore, "CORE"}, {IdbPortClass::kBump, "BUMP"}};

  _wire_state_list = {{IdbWiringStatement::kCover, "COVER"},
                      {IdbWiringStatement::kFixed, "FIXED"},
                      {IdbWiringStatement::kRouted, "ROUTED"},
                      {IdbWiringStatement::kNoShield, "NOSHIELD"},
                      {IdbWiringStatement::kShield, "SHIELD"}};

  _wire_shape_list = {{IdbWireShapeType::kRing, "RING"},
                      {IdbWireShapeType::kPadRing, "PADRING"},
                      {IdbWireShapeType::kBlockRing, "BLOCKRING"},
                      {IdbWireShapeType::kStripe, "STRIPE"},
                      {IdbWireShapeType::kFollowPin, "FOLLOWPIN"},
                      {IdbWireShapeType::kIoWire, "IOWIRE"},
                      {IdbWireShapeType::kCoreWire, "COREWIRE"},
                      {IdbWireShapeType::kBlockWire, "BLOCKWIRE"},
                      {IdbWireShapeType::kBlockageWire, "BLOCKAGEWIRE"},
                      {IdbWireShapeType::kFillWire, "FILLWIRE"},
                      {IdbWireShapeType::kFillWireOpc, "FILLWIREOPC"},
                      {IdbWireShapeType::kDrcFill, "DRCFILL"}};
}

IdbConnectDirection IdbConnectProperty::get_direction(string name)
{
  auto result = std::find_if(_direction_list.begin(), _direction_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _direction_list.end()) {
    return IdbConnectDirection::kNone;
  }

  return result->first;
}

string IdbConnectProperty::get_direction_name(IdbConnectDirection direction)
{
  auto iter = _direction_list.find(direction);
  if (iter == _direction_list.end()) {
    return string("");
  }

  return iter->second;
}

IdbConnectType IdbConnectProperty::get_type(string name)
{
  auto result = std::find_if(_type_list.begin(), _type_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _type_list.end()) {
    return IdbConnectType::kNone;
    ;
  }

  return result->first;
}

bool IdbConnectProperty::is_net(std::string name)
{
  auto type = get_type(name);
  return type == IdbConnectType::kSignal || type == IdbConnectType::kClock;
}

bool IdbConnectProperty::is_pdn(std::string name)
{
  auto type = get_type(name);
  return type == IdbConnectType::kPower || type == IdbConnectType::kGround;
}

string IdbConnectProperty::get_type_name(IdbConnectType type)
{
  auto iter = _type_list.find(type);
  if (iter == _type_list.end()) {
    return "";
  }

  return iter->second;
}

IdbTermShape IdbConnectProperty::get_pin_shape(string name)
{
  auto result = std::find_if(_pin_shape_list.begin(), _pin_shape_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _pin_shape_list.end()) {
    return IdbTermShape::kNone;
    ;
  }

  return result->first;
}

string IdbConnectProperty::get_pin_shape_name(IdbTermShape type)
{
  auto iter = _pin_shape_list.find(type);
  if (iter == _pin_shape_list.end()) {
    return "";
  }

  return iter->second;
}

IdbPortClass IdbConnectProperty::get_port_class(string name)
{
  auto result = std::find_if(_port_class_list.begin(), _port_class_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _port_class_list.end()) {
    return IdbPortClass::kNone;
    ;
  }

  return result->first;
}

string IdbConnectProperty::get_port_class_name(IdbPortClass type)
{
  auto iter = _port_class_list.find(type);
  if (iter == _port_class_list.end()) {
    return "";
  }

  return iter->second;
}

IdbWiringStatement IdbConnectProperty::get_wiring_state(string name)
{
  auto result = std::find_if(_wire_state_list.begin(), _wire_state_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _wire_state_list.end()) {
    return IdbWiringStatement::kNone;
    ;
  }

  return result->first;
}

string IdbConnectProperty::get_wiring_state_name(IdbWiringStatement type)
{
  auto iter = _wire_state_list.find(type);
  if (iter == _wire_state_list.end()) {
    return "";
  }

  return iter->second;
}

IdbWireShapeType IdbConnectProperty::get_wire_shape(string name)
{
  auto result = std::find_if(_wire_shape_list.begin(), _wire_shape_list.end(), [name](const auto& iter) { return iter.second == name; });

  if (result == _wire_shape_list.end()) {
    return IdbWireShapeType::kNone;
    ;
  }

  return result->first;
}

string IdbConnectProperty::get_wire_shape_name(IdbWireShapeType type)
{
  auto iter = _wire_shape_list.find(type);
  if (iter == _wire_shape_list.end()) {
    return "";
  }

  return iter->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbRegionProperty::IdbRegionProperty()
{
  _region_type_list = {{IdbRegionType::kFence, "FENCE"}, {IdbRegionType::kGuide, "GUIDE"}};
}

IdbRegionType IdbRegionProperty::get_type(string type_name)
{
  auto result = std::find_if(_region_type_list.begin(), _region_type_list.end(),
                             [type_name](const auto& iter) { return iter.second == type_name; });

  if (result == _region_type_list.end()) {
    return IdbRegionType::kNone;
  }

  return result->first;
}

string IdbRegionProperty::get_name(IdbRegionType type)
{
  auto iter = _region_type_list.find(type);
  if (iter == _region_type_list.end()) {
    return string("");
  }

  return iter->second;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbLayerProperty::IdbLayerProperty()
{
  _track_direction_list = {{IdbTrackDirection::kDirectionX, "X"}, {IdbTrackDirection::kDirectionY, "Y"}};

  _layer_type_list = {{IdbLayerType::kLayerCut, "CUT"},
                      {IdbLayerType::kLayerImplant, "IMPLANT"},
                      {IdbLayerType::kLayerMasterslice, "MASTERSLICE"},
                      {IdbLayerType::kLayerOverlap, "OVERLAP"},
                      {IdbLayerType::kLayerRouting, "ROUTING"}};

  _layer_direction_list = {{IdbLayerDirection::kHorizontal, "HORIZONTAL"},
                           {IdbLayerDirection::kVertical, "VERTICAL"},
                           {IdbLayerDirection::kDiag45, "DIAG45"},
                           {IdbLayerDirection::kDiag135, "DIAG135"}};
}

IdbTrackDirection IdbLayerProperty::get_track_direction(string direction_name)
{
  auto result = std::find_if(_track_direction_list.begin(), _track_direction_list.end(),
                             [direction_name](const auto& iter) { return iter.second == direction_name; });

  if (result == _track_direction_list.end()) {
    return IdbTrackDirection::kNone;
  }

  return result->first;
}

string IdbLayerProperty::get_track_direction_name(IdbTrackDirection direction)
{
  auto iter = _track_direction_list.find(direction);
  if (iter == _track_direction_list.end()) {
    return string("");
  }

  return iter->second;
}

IdbLayerType IdbLayerProperty::get_type(string type_name)
{
  auto result
      = std::find_if(_layer_type_list.begin(), _layer_type_list.end(), [type_name](const auto& iter) { return iter.second == type_name; });

  if (result == _layer_type_list.end()) {
    return IdbLayerType::kNone;
  }

  return result->first;
}

string IdbLayerProperty::get_name(IdbLayerType type)
{
  auto iter = _layer_type_list.find(type);
  if (iter == _layer_type_list.end()) {
    return string("");
  }

  return iter->second;
}

IdbLayerDirection IdbLayerProperty::get_direction(string direction_name)
{
  auto result = std::find_if(_layer_direction_list.begin(), _layer_direction_list.end(),
                             [direction_name](const auto& iter) { return iter.second == direction_name; });

  if (result == _layer_direction_list.end()) {
    return IdbLayerDirection::kNone;
  }

  return result->first;
}

string IdbLayerProperty::get_direction_str(IdbLayerDirection direction)
{
  auto iter = _layer_direction_list.find(direction);
  if (iter == _layer_direction_list.end()) {
    return string("");
  }

  return iter->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbCellProperty::IdbCellProperty()
{
  _type_list = {{CellMasterType::kNone, "NONE"},
                {CellMasterType::kCover, "COVER"},
                {CellMasterType::kCoverBump, "COVER BUMP"},
                {CellMasterType::kRing, "RING"},
                {CellMasterType::kBlock, "BLOCK"},
                {CellMasterType::kBlockBlackbox, "BLOCK BLACKBOX"},
                {CellMasterType::kBLockSoft, "BLOCK SOFT"},
                {CellMasterType::kPad, "PAD"},
                {CellMasterType::kPadInput, "PAD INPUT"},
                {CellMasterType::kPadOutput, "PAD OUTPUT"},
                {CellMasterType::kPadInOut, "PAD INOUT"},
                {CellMasterType::kPadPower, "PAD POWER"},
                {CellMasterType::kPadSpacer, "PAD SPACER"},
                {CellMasterType::kPadAreaIO, "PAD AREAIO"},
                {CellMasterType::kCore, "CORE"},
                {CellMasterType::kCoreFeedThru, "CORE FEEDTHRU"},
                {CellMasterType::kCoreTieHigh, "CORE TIEHIGH"},
                {CellMasterType::kCoreTieLow, "CORE TIELOW"},
                {CellMasterType::kCoreSpacer, "CORE SPACER"},
                {CellMasterType::kCoreAntenaCell, "CORE ANTENNACELL"},
                {CellMasterType::kCoreWelltap, "CORE WELLTAP"},
                {CellMasterType::kEndcap, "ENDCAP"},
                {CellMasterType::kEndcapPre, "ENDCAP PRE"},
                {CellMasterType::kEndcapPost, "ENDCAP POST"},
                {CellMasterType::kEndcapTopLeft, "ENDCAP TOPLEFT"},
                {CellMasterType::kEndcapTopRight, "ENDCAP TOPRIGHT"},
                {CellMasterType::kEndcapBottomLeft, "ENDCAP BOTTOMLEFT"},
                {CellMasterType::kEndcapBottomRight, "ENDCAP BOTTOMRIGHT"}};
}

CellMasterType IdbCellProperty::get_type(string type_name)
{
  auto result = std::find_if(_type_list.begin(), _type_list.end(), [type_name](const auto& iter) { return iter.second == type_name; });

  if (result == _type_list.end()) {
    return CellMasterType::kNone;
  }

  return result->first;
}

string IdbCellProperty::get_name(CellMasterType type)
{
  auto iter = _type_list.find(type);
  if (iter == _type_list.end()) {
    return string("");
  }

  return iter->second;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace idb
