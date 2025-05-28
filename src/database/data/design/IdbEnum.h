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
 * @file		IdbEnum.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Idb enum information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class IdbInstanceType : uint8_t
{
  kNone = 0,
  kNetlist,
  kDist,
  kUser,
  kTiming,
  kTest,
  kMax
};
#define STR_INSTANCE_TYPE_DIST "DIST"
#define STR_INSTANCE_TYPE_NETLIST "NETLIST"
#define STR_INSTANCE_TYPE_TIMING "TIMING"
#define STR_INSTANCE_TYPE_USER "USER"

enum class IdbPlacementStatus : uint8_t
{
  kNone = 0,
  kFixed,
  kCover,
  kPlaced,
  kUnplaced,
  kMax
};
#define STR_PLACE_STATUS_FIXED "FIXED"
#define STR_PLACE_STATUS_PLACED "PLACED"
#define STR_PLACE_STATUS_UNPLACED "UNPLACED"

// Placement status for the component.
// Default is 0
#define DEFI_COMPONENT_UNPLACED 1
#define DEFI_COMPONENT_PLACED 2
#define DEFI_COMPONENT_FIXED 3
#define DEFI_COMPONENT_COVER 4

class IdbInstancePropertyMap
{
 public:
  IdbInstancePropertyMap();
  ~IdbInstancePropertyMap() = default;

  IdbInstanceType get_type(std::string type_name);
  std::string get_type_str(IdbInstanceType type);

  IdbPlacementStatus get_status(std::string status_name);
  std::string get_status_str(IdbPlacementStatus status);

  IdbPlacementStatus get_status(int32_t status_def_enum);
  int32_t get_status_enum(IdbPlacementStatus status);

 private:
  std::map<IdbInstanceType, std::string> _type_list;
  std::map<IdbPlacementStatus, std::string> _status_list;
  std::map<IdbPlacementStatus, int32_t> _status_enum_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum class IdbSiteClass : uint8_t
{
  kNone = 0,
  kCore,
  kPad,
  kMax
};

enum class IdbSiteType : uint8_t
{
  kNone = 0,
  kCore,
  kPad,
  kCorner,
  kMax
};

enum class IdbSymmetry : uint8_t
{
  kNone = 0,
  kX,
  kY,
  kR90,
  kMax
};

/**
 * @brief Orientation
 * @reference  LEF/DEF 5.8 Language Reference P716
 * Maps the orientation terminology used in LEF and DEF files to the OpenAccess database format.
 *
 */
enum class IdbOrient : uint8_t
{
  kNone,
  kN_R0,
  kW_R90,
  kS_R180,
  kE_R270,
  kFN_MY,
  kFE_MY90,
  kFS_MX,
  kFW_MX90,
  kMax
};
#define STR_ORIENT_N "N"
#define STR_ORIENT_S "S"
#define STR_ORIENT_W "W"
#define STR_ORIENT_E "E"
#define STR_ORIENT_FN "FN"
#define STR_ORIENT_FS "FS"
#define STR_ORIENT_FW "FW"
#define STR_ORIENT_FE "FE"
/**
 * @brief the definition get from def 3rd party code
 */
#define DEF_ORIENT_N 0
#define DEF_ORIENT_W 1
#define DEF_ORIENT_S 2
#define DEF_ORIENT_E 3
#define DEF_ORIENT_FN 4
#define DEF_ORIENT_FW 5
#define DEF_ORIENT_FS 6
#define DEF_ORIENT_FE 7

class IdbSiteProperty
{
 public:
  IdbSiteProperty();
  ~IdbSiteProperty() = default;

  IdbSiteClass get_class_type(std::string class_name);
  std::string get_class_name(IdbSiteClass class_type);

  int32_t get_orient_def_value(IdbOrient idb_value);
  IdbOrient get_orient_idb_value(int32_t def_value);

  IdbOrient get_orient_value(std::string orient_name);
  std::string get_orient_name(IdbOrient oreint_value);

 private:
  std::map<IdbSiteClass, std::string> _site_class_list;
  std::map<int32_t, IdbOrient> _orient_list;
  std::map<IdbOrient, std::string> _orient_string_list;
  std::map<IdbOrient, std::string> _orient_alias_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class IdbPortClass : uint8_t
{
  kNone,
  kCore,
  kBump,
  kMax
};

enum class IdbConnectDirection : uint8_t
{
  kNone,
  kInput,
  kOutput,
  kOutputTriState,
  kInOut,
  kFeedThru,
  kMax
};

enum class IdbConnectType : uint8_t
{
  kNone = 0,
  kSignal = 1,
  kAnalog = 2,
  kPower = 3,
  kGround = 4,
  kClock = 5,
  kTieOff = 6,
  kScan = 7,
  kReset = 8,
  kMax
};

enum class IdbTermShape : uint8_t
{
  kNone,
  kAbutment,
  kRing,
  kFeedThru,
  kMax
};

enum class IdbWiringStatement : uint8_t
{
  kNone,
  kCover,
  kFixed,
  kRouted,
  kNoShield,
  kShield,
  kMax
};

enum class IdbWireShapeType : uint8_t
{
  kNone,
  kRing,
  kPadRing,
  kBlockRing,
  kStripe,
  kFollowPin,
  kIoWire,
  kCoreWire,
  kBlockWire,
  kBlockageWire,
  kFillWire,
  kFillWireOpc,
  kDrcFill,
  kMax,
};

class IdbConnectProperty
{
 public:
  IdbConnectProperty();
  ~IdbConnectProperty() = default;

  IdbConnectDirection get_direction(std::string name);
  std::string get_direction_name(IdbConnectDirection direction);

  IdbConnectType get_type(std::string name);
  std::string get_type_name(IdbConnectType type);

  IdbTermShape get_pin_shape(std::string name);
  std::string get_pin_shape_name(IdbTermShape type);

  IdbPortClass get_port_class(std::string name);
  std::string get_port_class_name(IdbPortClass type);

  IdbWiringStatement get_wiring_state(std::string name);
  std::string get_wiring_state_name(IdbWiringStatement type);

  IdbWireShapeType get_wire_shape(std::string name);
  std::string get_wire_shape_name(IdbWireShapeType type);

  bool is_net(std::string name);
  bool is_pdn(std::string name);

 private:
  std::map<IdbConnectDirection, std::string> _direction_list;
  std::map<IdbConnectType, std::string> _type_list;
  std::map<IdbTermShape, std::string> _pin_shape_list;
  std::map<IdbPortClass, std::string> _port_class_list;
  std::map<IdbWiringStatement, std::string> _wire_state_list;
  std::map<IdbWireShapeType, std::string> _wire_shape_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class IdbRegionType : uint8_t
{
  kNone,
  kFence,
  kGuide,
  kMax
};

class IdbRegionProperty
{
 public:
  IdbRegionProperty();
  ~IdbRegionProperty() = default;

  IdbRegionType get_type(std::string type_name);
  std::string get_name(IdbRegionType type);

 private:
  std::map<IdbRegionType, std::string> _region_type_list;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Specifies the location and direction of the first track defined. X indicates vertical lines; Y
indicates horizontal lines. start is the X or Y coordinate of the first line.
*/
enum class IdbTrackDirection : uint8_t
{
  kNone = 0,
  kDirectionX,
  kDirectionY,
  kMax
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class IdbLayerType : uint8_t
{
  kNone = 0,
  kLayerCut = 1,
  kLayerImplant = 2,
  kLayerMasterslice = 3,
  kLayerOverlap = 4,
  kLayerRouting = 5,
  kMax
};

enum class IdbLayerDirection : uint8_t
{
  kNone,
  kHorizontal,  // Routing parallel to the x axis is preferred
  kVertical,    // Routing parallel to the y axis is preferred
  kDiag45,      // Routing along a 45-degree angle is preferred
  kDiag135,     // Routing along a 135-degree angle is preferred
  kMax
};

// lefdefref page 455
// OFFSET {distance | xDistance yDistance}
// PITCH {distance | xDistance yDistance}
enum class IdbLayerOrientType : uint8_t
{
  kNone,
  kBothXY,      // Specifies one value that is used for both the x and y direction.
  kSeperateXY,  // Specifies the x for vertical routing tracks, and the y for horizontal routing tracks
  kMax
};

struct IdbLayerOrientValue
{
  IdbLayerOrientType type;
  int32_t orient_x;
  int32_t orient_y;
};

/*
 *Spacing rule in lefdefref page 458
 */
enum class IdbLayerSpacingType : uint8_t
{
  kNone,
  kSpacingDefault,
  kSpacingRange,
  kSpacingRangeLenThreshold,
  //!-----tbd-------------
  kMax
};

class IdbLayerProperty
{
 public:
  IdbLayerProperty();
  ~IdbLayerProperty() = default;

  IdbTrackDirection get_track_direction(std::string direction_name);
  std::string get_track_direction_name(IdbTrackDirection direction);

  IdbLayerType get_type(std::string type_name);
  std::string get_name(IdbLayerType type);

  IdbLayerDirection get_direction(std::string direction_name);
  std::string get_direction_str(IdbLayerDirection direction);

 private:
  std::map<IdbTrackDirection, std::string> _track_direction_list;
  std::map<IdbLayerType, std::string> _layer_type_list;
  std::map<IdbLayerDirection, std::string> _layer_direction_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum class CellMasterType : uint8_t
{
  kNone = 0,
  kCover,
  kCoverBump,
  kRing,
  kBlock,
  kBlockBlackbox,
  kBLockSoft,
  kPad,
  kPadInput,
  kPadOutput,
  kPadInOut,
  kPadPower,
  kPadSpacer,
  kPadAreaIO,
  kCore,
  kCoreFeedThru,
  kCoreTieHigh,
  kCoreTieLow,
  kCoreSpacer,
  kCoreAntenaCell,
  kCoreWelltap,
  kEndcap,
  kEndcapPre,
  kEndcapPost,
  kEndcapTopLeft,
  kEndcapTopRight,
  kEndcapBottomLeft,
  kEndcapBottomRight,
  kMax
};

class IdbCellProperty
{
 public:
  IdbCellProperty();
  ~IdbCellProperty() = default;

  CellMasterType get_type(std::string type_name);
  std::string get_name(CellMasterType type);

 private:
  std::map<CellMasterType, std::string> _type_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbEnum
{
 public:
  ~IdbEnum();

  static IdbEnum* GetInstance()
  {
    if (_instance == nullptr) {
      _mutex.lock();
      if (_instance == nullptr) {
        _instance = new IdbEnum();
      }
      _mutex.unlock();
    }
    return _instance;
  }

  IdbInstancePropertyMap* get_instance_property() { return _property_map; }
  IdbSiteProperty* get_site_property() { return _site_property; }
  IdbConnectProperty* get_connect_property() { return _term_property; }
  IdbLayerProperty* get_layer_property() { return _layer_property; }
  IdbCellProperty* get_cell_property() { return _cell_property; }
  IdbRegionProperty* get_region_property() { return _region_property; }

 private:
  IdbEnum();
  IdbEnum(IdbEnum& other) = delete;
  void operator=(const IdbEnum&) = delete;

  static IdbEnum* _instance;
  static std::mutex _mutex;

  IdbInstancePropertyMap* _property_map;
  IdbSiteProperty* _site_property;
  IdbConnectProperty* _term_property;
  IdbLayerProperty* _layer_property;
  IdbCellProperty* _cell_property;
  IdbRegionProperty* _region_property;
};

}  // namespace idb
