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
 * @file		def_read.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a def builder to build data structure from def.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "def_read.h"

#include <regex>

#include "../../../data/design/IdbDesign.h"
#include "../def/defzlib/defzlib.hpp"
#include "Str.hh"
#include "defiPath.hpp"
#include "defrReader.hpp"

using std::cout;
using std::endl;

namespace idb {

DefRead::DefRead(IdbDefService* def_service)
{
  _def_service = def_service;
  _cur_cell_master = nullptr;
}

DefRead::~DefRead()
{
}

bool DefRead::check_type(defrCallbackType_e type)
{
  if (type >= 0 && type <= defrDesignEndCbkType) {
    return true;
  } else {
    std::cout << "Error defrCallbackType_e = " << type << std::endl;
    return false;
  }
}

bool DefRead::createDb(const char* file)
{
  if (ieda::Str::contain(file, ".gz")) {
    return createDbGzip(file);
  } else {
    FILE* f = fopen(file, "r");

    if (f == NULL) {
      std::cerr << "Open def file failed..." << std::endl;
      return false;
    }

    defrInit();
    defrReset();

    defrInitSession();
    defrSetVersionStrCbk(versionCallback);
    defrSetDesignCbk(designCallback);
    defrSetBusBitCbk(busBitCharsCallBack);
    //   defrSetPropCbk(propCallback);
    //   defrSetPropDefEndCbk(propEndCallback);
    //   defrSetPropDefStartCbk(propStartCallback);
    //  defrSetBlockageStartCbk(blockageBeginCallback);
    defrSetBlockageCbk(blockageCallback);
    //  defrSetBlockageEndCbk(blockageEndCallback);
    defrSetComponentCbk(componentsCallback);
    defrSetComponentStartCbk(componentNumberCallback);
    defrSetComponentEndCbk(componentEndCallback);
    //   defrSetComponentMaskShiftLayerCbk(componentMaskShiftCallback);
    //   defrSetExtensionCbk(extensionCallback);
    defrSetFillStartCbk(fillsCallback);
    defrSetFillCbk(fillCallback);
    defrSetGcellGridCbk(gcellGridCallback);
    defrSetGroupCbk(groupCallback);
    //   defrSetGroupMemberCbk(groupMemberCallback);
    //   defrSetGroupNameCbk(groupNameCallback);
    //   defrSetHistoryCbk(historyCallback);
    defrSetNetStartCbk(netBeginCallback);
    defrSetNetCbk(netCallback);
    defrSetNetEndCbk(netEndCallback);
    //   defrSetNonDefaultCbk(nonDefaultRuleCallback);
    defrSetPinCbk(pinCallback);
    defrSetPinEndCbk(pinsEndCallback);
    defrSetStartPinsCbk(pinsBeginCallback);
    //   defrSetPinPropCbk(pinPropCallback);
    defrSetRegionCbk(regionCallback);
    defrSetRowCbk(rowCallback);
    //   defrSetScanchainsStartCbk(scanchainsCallback);
    defrSetSlotCbk(slotsCallback);
    defrSetSNetStartCbk(specialNetBeginCallback);
    defrSetSNetCbk(specialNetCallback);
    defrSetSNetEndCbk(specialNetEndCallback);
    // defrSetStartPinsCbk(pinsStartCallback);
    //   defrSetStylesStartCbk(stylesCallback);
    //   defrSetTechnologyCbk(technologyCallback);
    defrSetUnitsCbk(unitsCallback);
    defrSetViaCbk(viaCallback);
    defrSetViaStartCbk(viaBeginCallback);

    defrSetAddPathToNet();
    defrSetDieAreaCbk(dieAreaCallback);
    defrSetTrackCbk(trackGridCallback);
    // void* userData = (void*) 0x01020304;

    int res = defrRead(f, file, (defiUserData) this, /* case sensitive */ 1);

    if (res != 0) {
      return false;
    }

    (void) defrUnsetCallbacks();

    // Unset all the callbacks
    defrUnsetArrayNameCbk();
    defrUnsetAssertionCbk();
    defrUnsetAssertionsStartCbk();
    defrUnsetAssertionsEndCbk();
    defrUnsetBlockageCbk();
    defrUnsetBlockageStartCbk();
    defrUnsetBlockageEndCbk();
    defrUnsetBusBitCbk();
    defrUnsetCannotOccupyCbk();
    defrUnsetCanplaceCbk();
    defrUnsetCaseSensitiveCbk();
    defrUnsetComponentCbk();
    defrUnsetComponentExtCbk();
    defrUnsetComponentStartCbk();
    defrUnsetComponentEndCbk();
    defrUnsetConstraintCbk();
    defrUnsetConstraintsStartCbk();
    defrUnsetConstraintsEndCbk();
    defrUnsetDefaultCapCbk();
    defrUnsetDesignCbk();
    defrUnsetDesignEndCbk();
    defrUnsetDieAreaCbk();
    defrUnsetDividerCbk();
    defrUnsetExtensionCbk();
    defrUnsetFillCbk();
    defrUnsetFillStartCbk();
    defrUnsetFillEndCbk();
    defrUnsetFPCCbk();
    defrUnsetFPCStartCbk();
    defrUnsetFPCEndCbk();
    defrUnsetFloorPlanNameCbk();
    defrUnsetGcellGridCbk();
    defrUnsetGroupCbk();
    defrUnsetGroupExtCbk();
    defrUnsetGroupMemberCbk();
    defrUnsetComponentMaskShiftLayerCbk();
    defrUnsetGroupNameCbk();
    defrUnsetGroupsStartCbk();
    defrUnsetGroupsEndCbk();
    defrUnsetHistoryCbk();
    defrUnsetIOTimingCbk();
    defrUnsetIOTimingsStartCbk();
    defrUnsetIOTimingsEndCbk();
    defrUnsetIOTimingsExtCbk();
    defrUnsetNetCbk();
    defrUnsetNetNameCbk();
    defrUnsetNetNonDefaultRuleCbk();
    defrUnsetNetConnectionExtCbk();
    defrUnsetNetExtCbk();
    defrUnsetNetPartialPathCbk();
    defrUnsetNetSubnetNameCbk();
    defrUnsetNetStartCbk();
    defrUnsetNetEndCbk();
    defrUnsetNonDefaultCbk();
    defrUnsetNonDefaultStartCbk();
    defrUnsetNonDefaultEndCbk();
    defrUnsetPartitionCbk();
    defrUnsetPartitionsExtCbk();
    defrUnsetPartitionsStartCbk();
    defrUnsetPartitionsEndCbk();
    defrUnsetPathCbk();
    defrUnsetPinCapCbk();
    defrUnsetPinCbk();
    defrUnsetPinEndCbk();
    defrUnsetPinExtCbk();
    defrUnsetPinPropCbk();
    defrUnsetPinPropStartCbk();
    defrUnsetPinPropEndCbk();
    defrUnsetPropCbk();
    defrUnsetPropDefEndCbk();
    defrUnsetPropDefStartCbk();
    defrUnsetRegionCbk();
    defrUnsetRegionStartCbk();
    defrUnsetRegionEndCbk();
    defrUnsetRowCbk();
    defrUnsetScanChainExtCbk();
    defrUnsetScanchainCbk();
    defrUnsetScanchainsStartCbk();
    defrUnsetScanchainsEndCbk();
    defrUnsetSiteCbk();
    defrUnsetSlotCbk();
    defrUnsetSlotStartCbk();
    defrUnsetSlotEndCbk();
    defrUnsetSNetWireCbk();
    defrUnsetSNetCbk();
    defrUnsetSNetStartCbk();
    defrUnsetSNetEndCbk();
    defrUnsetSNetPartialPathCbk();
    defrUnsetStartPinsCbk();
    defrUnsetStylesCbk();
    defrUnsetStylesStartCbk();
    defrUnsetStylesEndCbk();
    defrUnsetTechnologyCbk();
    defrUnsetTimingDisableCbk();
    defrUnsetTimingDisablesStartCbk();
    defrUnsetTimingDisablesEndCbk();
    defrUnsetTrackCbk();
    defrUnsetUnitsCbk();
    defrUnsetVersionCbk();
    defrUnsetVersionStrCbk();
    defrUnsetViaCbk();
    defrUnsetViaExtCbk();
    defrUnsetViaStartCbk();
    defrUnsetViaEndCbk();

    defrClear();

    fclose(f);

    return true;
  }
}

bool DefRead::createDbGzip(const char* gzip_file)
{
  defGZFile f = defrGZipOpen(gzip_file, "r");

  if (f == NULL) {
    std::cerr << "Open def file failed..." << std::endl;
    return false;
  }

  defrInit();
  defrReset();

  defrInitSession();
  defrSetGZipReadFunction();
  defrSetVersionStrCbk(versionCallback);
  defrSetDesignCbk(designCallback);
  defrSetBusBitCbk(busBitCharsCallBack);
  //   defrSetPropCbk(propCallback);
  //   defrSetPropDefEndCbk(propEndCallback);
  //   defrSetPropDefStartCbk(propStartCallback);
  //  defrSetBlockageStartCbk(blockageBeginCallback);
  defrSetBlockageCbk(blockageCallback);
  //  defrSetBlockageEndCbk(blockageEndCallback);
  defrSetComponentCbk(componentsCallback);
  defrSetComponentStartCbk(componentNumberCallback);
  defrSetComponentEndCbk(componentEndCallback);
  //   defrSetComponentMaskShiftLayerCbk(componentMaskShiftCallback);
  //   defrSetExtensionCbk(extensionCallback);
  defrSetFillStartCbk(fillsCallback);
  defrSetFillCbk(fillCallback);
  defrSetGcellGridCbk(gcellGridCallback);
  defrSetGroupCbk(groupCallback);
  //   defrSetGroupMemberCbk(groupMemberCallback);
  //   defrSetGroupNameCbk(groupNameCallback);
  //   defrSetHistoryCbk(historyCallback);
  defrSetNetStartCbk(netBeginCallback);
  defrSetNetCbk(netCallback);
  defrSetNetEndCbk(netEndCallback);
  //   defrSetNonDefaultCbk(nonDefaultRuleCallback);
  defrSetPinCbk(pinCallback);
  defrSetPinEndCbk(pinsEndCallback);
  defrSetStartPinsCbk(pinsBeginCallback);
  //   defrSetPinPropCbk(pinPropCallback);
  defrSetRegionCbk(regionCallback);
  defrSetRowCbk(rowCallback);
  //   defrSetScanchainsStartCbk(scanchainsCallback);
  defrSetSlotCbk(slotsCallback);
  defrSetSNetStartCbk(specialNetBeginCallback);
  defrSetSNetCbk(specialNetCallback);
  defrSetSNetEndCbk(specialNetEndCallback);
  // defrSetStartPinsCbk(pinsStartCallback);
  //   defrSetStylesStartCbk(stylesCallback);
  //   defrSetTechnologyCbk(technologyCallback);
  defrSetUnitsCbk(unitsCallback);
  defrSetViaCbk(viaCallback);
  defrSetViaStartCbk(viaBeginCallback);

  defrSetAddPathToNet();
  defrSetDieAreaCbk(dieAreaCallback);
  defrSetTrackCbk(trackGridCallback);
  // void* userData = (void*) 0x01020304;

  int res = defrReadGZip(f, gzip_file, (defiUserData) this);

  if (res != 0) {
    return false;
  }

  (void) defrUnsetCallbacks();

  // Unset all the callbacks
  defrUnsetArrayNameCbk();
  defrUnsetAssertionCbk();
  defrUnsetAssertionsStartCbk();
  defrUnsetAssertionsEndCbk();
  defrUnsetBlockageCbk();
  defrUnsetBlockageStartCbk();
  defrUnsetBlockageEndCbk();
  defrUnsetBusBitCbk();
  defrUnsetCannotOccupyCbk();
  defrUnsetCanplaceCbk();
  defrUnsetCaseSensitiveCbk();
  defrUnsetComponentCbk();
  defrUnsetComponentExtCbk();
  defrUnsetComponentStartCbk();
  defrUnsetComponentEndCbk();
  defrUnsetConstraintCbk();
  defrUnsetConstraintsStartCbk();
  defrUnsetConstraintsEndCbk();
  defrUnsetDefaultCapCbk();
  defrUnsetDesignCbk();
  defrUnsetDesignEndCbk();
  defrUnsetDieAreaCbk();
  defrUnsetDividerCbk();
  defrUnsetExtensionCbk();
  defrUnsetFillCbk();
  defrUnsetFillStartCbk();
  defrUnsetFillEndCbk();
  defrUnsetFPCCbk();
  defrUnsetFPCStartCbk();
  defrUnsetFPCEndCbk();
  defrUnsetFloorPlanNameCbk();
  defrUnsetGcellGridCbk();
  defrUnsetGroupCbk();
  defrUnsetGroupExtCbk();
  defrUnsetGroupMemberCbk();
  defrUnsetComponentMaskShiftLayerCbk();
  defrUnsetGroupNameCbk();
  defrUnsetGroupsStartCbk();
  defrUnsetGroupsEndCbk();
  defrUnsetHistoryCbk();
  defrUnsetIOTimingCbk();
  defrUnsetIOTimingsStartCbk();
  defrUnsetIOTimingsEndCbk();
  defrUnsetIOTimingsExtCbk();
  defrUnsetNetCbk();
  defrUnsetNetNameCbk();
  defrUnsetNetNonDefaultRuleCbk();
  defrUnsetNetConnectionExtCbk();
  defrUnsetNetExtCbk();
  defrUnsetNetPartialPathCbk();
  defrUnsetNetSubnetNameCbk();
  defrUnsetNetStartCbk();
  defrUnsetNetEndCbk();
  defrUnsetNonDefaultCbk();
  defrUnsetNonDefaultStartCbk();
  defrUnsetNonDefaultEndCbk();
  defrUnsetPartitionCbk();
  defrUnsetPartitionsExtCbk();
  defrUnsetPartitionsStartCbk();
  defrUnsetPartitionsEndCbk();
  defrUnsetPathCbk();
  defrUnsetPinCapCbk();
  defrUnsetPinCbk();
  defrUnsetPinEndCbk();
  defrUnsetPinExtCbk();
  defrUnsetPinPropCbk();
  defrUnsetPinPropStartCbk();
  defrUnsetPinPropEndCbk();
  defrUnsetPropCbk();
  defrUnsetPropDefEndCbk();
  defrUnsetPropDefStartCbk();
  defrUnsetRegionCbk();
  defrUnsetRegionStartCbk();
  defrUnsetRegionEndCbk();
  defrUnsetRowCbk();
  defrUnsetScanChainExtCbk();
  defrUnsetScanchainCbk();
  defrUnsetScanchainsStartCbk();
  defrUnsetScanchainsEndCbk();
  defrUnsetSiteCbk();
  defrUnsetSlotCbk();
  defrUnsetSlotStartCbk();
  defrUnsetSlotEndCbk();
  defrUnsetSNetWireCbk();
  defrUnsetSNetCbk();
  defrUnsetSNetStartCbk();
  defrUnsetSNetEndCbk();
  defrUnsetSNetPartialPathCbk();
  defrUnsetStartPinsCbk();
  defrUnsetStylesCbk();
  defrUnsetStylesStartCbk();
  defrUnsetStylesEndCbk();
  defrUnsetTechnologyCbk();
  defrUnsetTimingDisableCbk();
  defrUnsetTimingDisablesStartCbk();
  defrUnsetTimingDisablesEndCbk();
  defrUnsetTrackCbk();
  defrUnsetUnitsCbk();
  defrUnsetVersionCbk();
  defrUnsetVersionStrCbk();
  defrUnsetViaCbk();
  defrUnsetViaExtCbk();
  defrUnsetViaStartCbk();
  defrUnsetViaEndCbk();

  defrClear();

  defrGZipClose(f);

  return true;
}

bool DefRead::createFloorplanDb(const char* file)
{
  return createDb(file);

  FILE* f = fopen(file, "r");
  if (f == NULL) {
    std::cout << "Open def file failed..." << std::endl;
    return false;
  }

  defrInit();
  defrReset();

  defrInitSession();
  defrSetVersionStrCbk(versionCallback);
  defrSetDesignCbk(designCallback);
  //   defrSetPropCbk(propCallback);
  //   defrSetPropDefEndCbk(propEndCallback);
  //   defrSetPropDefStartCbk(propStartCallback);
  //  defrSetBlockageStartCbk(blockageBeginCallback);
  defrSetBlockageCbk(blockageCallback);
  //  defrSetBlockageEndCbk(blockageEndCallback);
  defrSetComponentCbk(componentsCallback);
  defrSetComponentStartCbk(componentNumberCallback);
  defrSetComponentEndCbk(componentEndCallback);
  //   defrSetComponentMaskShiftLayerCbk(componentMaskShiftCallback);
  //   defrSetExtensionCbk(extensionCallback);
  defrSetFillStartCbk(fillsCallback);
  defrSetFillCbk(fillCallback);
  defrSetGcellGridCbk(gcellGridCallback);
  defrSetGroupCbk(groupCallback);
  //   defrSetGroupMemberCbk(groupMemberCallback);
  //   defrSetGroupNameCbk(groupNameCallback);
  //   defrSetHistoryCbk(historyCallback);
  defrSetNetStartCbk(netBeginCallback);
  defrSetNetCbk(netCallback);
  defrSetNetEndCbk(netEndCallback);
  //   defrSetNonDefaultCbk(nonDefaultRuleCallback);
  defrSetPinCbk(pinCallback);
  defrSetPinEndCbk(pinsEndCallback);
  defrSetStartPinsCbk(pinsBeginCallback);
  //   defrSetPinPropCbk(pinPropCallback);
  defrSetRegionCbk(regionCallback);
  defrSetRowCbk(rowCallback);
  //   defrSetScanchainsStartCbk(scanchainsCallback);
  defrSetSlotCbk(slotsCallback);
  defrSetSNetStartCbk(specialNetBeginCallback);
  defrSetSNetCbk(specialNetCallback);
  defrSetSNetEndCbk(specialNetEndCallback);
  // defrSetStartPinsCbk(pinsStartCallback);
  //   defrSetStylesStartCbk(stylesCallback);
  //   defrSetTechnologyCbk(technologyCallback);
  defrSetUnitsCbk(unitsCallback);
  defrSetViaCbk(viaCallback);
  defrSetViaStartCbk(viaBeginCallback);

  defrSetAddPathToNet();
  defrSetDieAreaCbk(dieAreaCallback);
  defrSetTrackCbk(trackGridCallback);
  // void* userData = (void*) 0x01020304;
  int res = defrRead(f, file, (defiUserData) this, /* case sensitive */ 1);
  if (res != 0) {
    return false;
  }

  (void) defrUnsetCallbacks();

  // Unset all the callbacks
  defrUnsetArrayNameCbk();
  defrUnsetAssertionCbk();
  defrUnsetAssertionsStartCbk();
  defrUnsetAssertionsEndCbk();
  defrUnsetBlockageCbk();
  defrUnsetBlockageStartCbk();
  defrUnsetBlockageEndCbk();
  defrUnsetBusBitCbk();
  defrUnsetCannotOccupyCbk();
  defrUnsetCanplaceCbk();
  defrUnsetCaseSensitiveCbk();
  defrUnsetComponentCbk();
  defrUnsetComponentExtCbk();
  defrUnsetComponentStartCbk();
  defrUnsetComponentEndCbk();
  defrUnsetConstraintCbk();
  defrUnsetConstraintsStartCbk();
  defrUnsetConstraintsEndCbk();
  defrUnsetDefaultCapCbk();
  defrUnsetDesignCbk();
  defrUnsetDesignEndCbk();
  defrUnsetDieAreaCbk();
  defrUnsetDividerCbk();
  defrUnsetExtensionCbk();
  defrUnsetFillCbk();
  defrUnsetFillStartCbk();
  defrUnsetFillEndCbk();
  defrUnsetFPCCbk();
  defrUnsetFPCStartCbk();
  defrUnsetFPCEndCbk();
  defrUnsetFloorPlanNameCbk();
  defrUnsetGcellGridCbk();
  defrUnsetGroupCbk();
  defrUnsetGroupExtCbk();
  defrUnsetGroupMemberCbk();
  defrUnsetComponentMaskShiftLayerCbk();
  defrUnsetGroupNameCbk();
  defrUnsetGroupsStartCbk();
  defrUnsetGroupsEndCbk();
  defrUnsetHistoryCbk();
  defrUnsetIOTimingCbk();
  defrUnsetIOTimingsStartCbk();
  defrUnsetIOTimingsEndCbk();
  defrUnsetIOTimingsExtCbk();
  defrUnsetNetCbk();
  defrUnsetNetNameCbk();
  defrUnsetNetNonDefaultRuleCbk();
  defrUnsetNetConnectionExtCbk();
  defrUnsetNetExtCbk();
  defrUnsetNetPartialPathCbk();
  defrUnsetNetSubnetNameCbk();
  defrUnsetNetStartCbk();
  defrUnsetNetEndCbk();
  defrUnsetNonDefaultCbk();
  defrUnsetNonDefaultStartCbk();
  defrUnsetNonDefaultEndCbk();
  defrUnsetPartitionCbk();
  defrUnsetPartitionsExtCbk();
  defrUnsetPartitionsStartCbk();
  defrUnsetPartitionsEndCbk();
  defrUnsetPathCbk();
  defrUnsetPinCapCbk();
  defrUnsetPinCbk();
  defrUnsetPinEndCbk();
  defrUnsetPinExtCbk();
  defrUnsetPinPropCbk();
  defrUnsetPinPropStartCbk();
  defrUnsetPinPropEndCbk();
  defrUnsetPropCbk();
  defrUnsetPropDefEndCbk();
  defrUnsetPropDefStartCbk();
  defrUnsetRegionCbk();
  defrUnsetRegionStartCbk();
  defrUnsetRegionEndCbk();
  defrUnsetRowCbk();
  defrUnsetScanChainExtCbk();
  defrUnsetScanchainCbk();
  defrUnsetScanchainsStartCbk();
  defrUnsetScanchainsEndCbk();
  defrUnsetSiteCbk();
  defrUnsetSlotCbk();
  defrUnsetSlotStartCbk();
  defrUnsetSlotEndCbk();
  defrUnsetSNetWireCbk();
  defrUnsetSNetCbk();
  defrUnsetSNetStartCbk();
  defrUnsetSNetEndCbk();
  defrUnsetSNetPartialPathCbk();
  defrUnsetStartPinsCbk();
  defrUnsetStylesCbk();
  defrUnsetStylesStartCbk();
  defrUnsetStylesEndCbk();
  defrUnsetTechnologyCbk();
  defrUnsetTimingDisableCbk();
  defrUnsetTimingDisablesStartCbk();
  defrUnsetTimingDisablesEndCbk();
  defrUnsetTrackCbk();
  defrUnsetUnitsCbk();
  defrUnsetVersionCbk();
  defrUnsetVersionStrCbk();
  defrUnsetViaCbk();
  defrUnsetViaExtCbk();
  defrUnsetViaStartCbk();
  defrUnsetViaEndCbk();

  defrClear();

  fclose(f);

  return true;
}

int32_t DefRead::versionCallback(defrCallbackType_e type, const char* version, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Version] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_version(version);

  return kDbSuccess;
}

int32_t DefRead::parse_version(const char* version)
{
  IdbDesign* design = _def_service->get_design();
  design->set_version(version);

  return kDbSuccess;
}

int32_t DefRead::designCallback(defrCallbackType_e type, const char* name, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Design name] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_design(name);

  return kDbSuccess;
}

int32_t DefRead::parse_design(const char* name)
{
  IdbDesign* design = _def_service->get_design();
  design->set_design_name(name);

  return kDbSuccess;
}

int32_t DefRead::unitsCallback(defrCallbackType_e type, double d, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Units] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_units(d);

  return kDbSuccess;
}

int32_t DefRead::parse_units(double microns)
{
  IdbDesign* design = _def_service->get_design();
  IdbLayout* layout = _def_service->get_layout();

  uint32_t lef_microns = layout->get_units()->get_micron_dbu();
  if (microns != lef_microns) {
    std::cout << "Warning : Def DBU dismatch LEF DBU" << std::endl;
    //   return kDbFail;
  }

  IdbUnits* units = design->get_units();
  units->set_microns_dbu(microns);

  return kDbSuccess;
}

int32_t DefRead::dieAreaCallback(defrCallbackType_e type, defiBox* def_box, defiUserData data)
{
  if (def_box == nullptr) {
    std::cout << "Die Area is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Die Area] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_die(def_box);

  return kDbSuccess;
}

int32_t DefRead::parse_die(defiBox* def_box)
{
  if (def_box == nullptr) {
    std::cout << "Parse die error..." << std::endl;

    return kDbFail;
  }

  // IdbDesign* design = _def_service->get_design();
  IdbLayout* layout = _def_service->get_layout();
  IdbDie* die = layout->get_die();

  defiPoints points = def_box->getPoint();
  for (int i = 0; i < points.numPoints; ++i) {
    die->add_point(points.x[i], points.y[i]);
  }

  die->set_bounding_box();

  return kDbSuccess;
}

int32_t DefRead::trackGridCallback(defrCallbackType_e type, defiTrack* def_track, defiUserData data)
{
  if (def_track == nullptr) {
    std::cout << "Track Grid is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Track Grid] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_track_grid(def_track);

  return kDbSuccess;
}

int32_t DefRead::parse_track_grid(defiTrack* def_track)
{
  if (def_track == nullptr) {
    std::cout << "Track Grid is nullPtr..." << std::endl;
    return kDbFail;
  }

  // IdbDesign* design = _def_service->get_design(); // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbTrackGridList* track_grid_list = layout->get_track_grid_list();
  IdbTrackGrid* track_grid = track_grid_list->add_track_grid(nullptr);
  IdbTrack* track = track_grid->get_track();

  IdbTrackDirection direction = def_track->macro()[0] == 'X' ? IdbTrackDirection::kDirectionX : IdbTrackDirection::kDirectionY;
  track->set_direction(direction);
  track->set_start(def_track->x());
  track->set_pitch(def_track->xStep());
  // <<---tbd--->>
  // db_track->set_width();
  // db_track->set_layer();
  track_grid->set_track_number(def_track->xNum());
  // <<---tbd--->>
  IdbLayers* layers = layout->get_layers();
  for (int i = 0; i < def_track->numLayers(); ++i) {
    const char* layer_name = def_track->layer(i);
    IdbLayer* layer = layers->find_layer(layer_name);
    if (layer != nullptr) {
      track_grid->add_layer_list(layer);
      if (layer->is_routing()) {
        IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
        routing_layer->add_track_grid(track_grid);
      }
    } else {
      std::cout << "Track Grid Error : no layer exist..." << std::endl;
    }
  }

  return kDbSuccess;
}

int32_t DefRead::rowCallback(defrCallbackType_e type, defiRow* def_row, defiUserData data)
{
  if (def_row == nullptr) {
    std::cout << "Row is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Row] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_row(def_row);

  return kDbSuccess;
}

int32_t DefRead::parse_row(defiRow* def_row)
{
  if (def_row == nullptr) {
    std::cout << "Row is nullPtr..." << std::endl;
    return kDbFail;
  }

  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbRows* rows = layout->get_rows();
  IdbRow* row = rows->add_row_list(nullptr);

  row->set_name(def_row->name());
  row->set_original_coordinate(def_row->x(), def_row->y());

  IdbSites* sites = layout->get_sites();
  IdbSite* lef_site = sites->add_site_list(def_row->macro());
  IdbSite* row_site = lef_site->clone();
  row_site->set_orient_by_enum(def_row->orient());
  row->set_site(row_site);
  row->set_orient(row_site->get_orient());

  if (def_row->hasDo()) {
    row->set_row_num_x(def_row->xNum());
    row->set_row_num_y(def_row->yNum());
    if (def_row->hasDoStep()) {
      row->set_step_x(def_row->xStep());
      row->set_step_y(def_row->yStep());
    }
  }

  row->set_bounding_box();

  // std::cout << "Parse row success..." << std::endl;
  return kDbSuccess;
}

int32_t DefRead::componentNumberCallback(defrCallbackType_e type, int def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Component] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_component_number(def_num);
  def_reader->set_start_time(clock());

  return kDbSuccess;
}

int32_t DefRead::parse_component_number(int32_t def_component_num)
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbInstanceList* instance_list = design->get_instance_list();
  instance_list->init(def_component_num);

  return kDbSuccess;
}

int32_t DefRead::componentsCallback(defrCallbackType_e type, defiComponent* def_component, defiUserData data)
{
  if (def_component == nullptr) {
    std::cout << "Component is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Component] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_component(def_component);

  return kDbSuccess;
}

int32_t DefRead::parse_component(defiComponent* def_component)
{
  if (def_component == nullptr) {
    std::cout << "Component is nullPtr..." << std::endl;
    return kDbFail;
  }

  IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();
  IdbRegionList* region_list = design->get_region_list();
  IdbInstanceList* instance_list = design->get_instance_list();
  IdbCellMasterList* master_list = layout->get_cell_master_list();

  if (nullptr == _cur_cell_master || _cur_cell_master->get_name() != def_component->name()) {
    _cur_cell_master = master_list->find_cell_master(def_component->name());
  }
  if (_cur_cell_master == nullptr) {
    std::cout << "Error can not find Cell Master : " << def_component->name() << std::endl;
    return kDbFail;
  }

  std::string inst_name = def_component->id();
  std::string new_inst_name = ieda::Str::trimEscape(inst_name);

  IdbInstance* instance = instance_list->add_instance(new_inst_name);
  if (instance == nullptr) {
    std::cout << "Create Instance Error..." << std::endl;
    return kDbFail;
  }
  instance->set_cell_master(_cur_cell_master);
  instance->set_status_by_def_enum(def_component->placementStatus());
  instance->set_orient_by_enum(def_component->placementOrient());

  if (def_component->hasSource()) {
    instance->set_type(def_component->source());
  }

  if (def_component->hasWeight()) {
    instance->set_weight(def_component->weight());
  }

  if (def_component->hasRegionName()) {
    IdbRegion* region = region_list->find_region(def_component->regionName());
    if (region != nullptr) {
      instance->set_region(region);
      region->add_instance(instance);
    }
  }

  if (def_component->hasHalo()) {
    IdbHalo* halo = instance->set_halo();
    halo->set_soft(def_component->hasHaloSoft());
    int32_t extend_left, extend_right, extend_top, extend_bottom;
    def_component->haloEdges(&extend_left, &extend_bottom, &extend_right, &extend_top);

    halo->set_extend_lef(extend_left);
    halo->set_extend_right(extend_right);
    halo->set_extend_bottom(extend_bottom);
    halo->set_extend_top(extend_top);
  }

  if (def_component->hasRouteHalo()) {
    IdbRouteHalo* route_halo = instance->set_route_halo();
    route_halo->set_route_distance(def_component->haloDist());
    route_halo->set_layer_bottom(layer_list->find_layer(def_component->minLayer()));
    route_halo->set_layer_top(layer_list->find_layer(def_component->maxLayer()));
  }

  instance->set_coodinate(def_component->placementX(), def_component->placementY());

  if (instance_list->get_num() % 1000 == 0) {
    std::cout << "-" << std::flush;
    if (instance_list->get_num() % 100000 == 0) {
      std::cout << std::endl;
    }
  }

  /// clear def_component
  def_component->clear();
  def_component->setPlacementLocation(0, 0, 0);

  return kDbSuccess;
}

int32_t DefRead::componentEndCallback(defrCallbackType_e type, void*, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Component] ..." << std::endl;
    return kDbFail;
  }

  std::cout << std::endl;
  def_reader->set_end_time(clock());

  return kDbSuccess;
}

int32_t DefRead::netBeginCallback(defrCallbackType_e type, int def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Net] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_net_number(def_num);
  def_reader->set_start_time(clock());

  return kDbSuccess;
}

int32_t DefRead::parse_net_number(int32_t def_net_num)
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbNetList* net_list = design->get_net_list();
  net_list->init(def_net_num);

  return kDbSuccess;
}

int32_t DefRead::netCallback(defrCallbackType_e type, defiNet* def_net, defiUserData data)
{
  if (def_net == nullptr) {
    std::cout << "Net is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Net] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_net(def_net);

  return kDbSuccess;
}

int32_t DefRead::parse_net(defiNet* def_net)
{
  if (def_net == nullptr) {
    std::cout << "Net is nullPtr..." << std::endl;
    return kDbFail;
  }

  IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();

  IdbPins* io_pin_list = design->get_io_pin_list();
  IdbInstanceList* instance_list = design->get_instance_list();

  IdbNetList* net_list = design->get_net_list();

  //   IdbNet* net = net_list->add_net(def_net->name());

  std::string net_name = def_net->name();
  std::string new_net_name = ieda::Str::trimEscape(net_name);
  IdbNet* net = net_list->add_net(new_net_name);

  if (net == nullptr) {
    std::cout << "Create Net Error..." << std::endl;
    return kDbFail;
  }

  if (def_net->hasUse()) {
    net->set_connect_type(def_net->use());
  }

  if (def_net->hasSource()) {
    net->set_source_type(def_net->source());
  }

  if (def_net->hasWeight()) {
    net->set_weight(def_net->weight());
  }

  if (def_net->hasXTalk()) {
    net->set_xtalk(def_net->XTalk());
  }

  if (def_net->hasFrequency()) {
    net->set_frequency(def_net->frequency());
  }

  if (def_net->hasOriginal()) {
    net->set_original_net_name(def_net->original());
  }

  int num_connections = def_net->numConnections();
  auto setPinNet = [net, num_connections](IdbPin* pin) {
    if (num_connections < 2) {
      if (pin->get_net() == nullptr) {
        pin->set_net(net);
      }
    } else {
      pin->set_net(net);
    }
  };

  for (int i = 0; i < num_connections; i++) {
    std::string io_name = def_net->instance(i);
    io_name = ieda::Str::trimEscape(io_name);

    IdbPin* pin = nullptr;
    if (io_name.compare("PIN") == 0) {
      std::string pin_name = def_net->pin(i);
      pin_name = ieda::Str::trimEscape(pin_name);
      pin = io_pin_list->find_pin(pin_name);
      if (pin == nullptr) {
        std::cout << "Can not find Pin in Pin list ... pin name = " << def_net->pin(i) << std::endl;
      } else {
        net->add_io_pin(pin);
        setPinNet(pin);
      }
    } else {
      IdbInstance* instance = instance_list->find_instance(io_name);
      if (instance != nullptr) {
        net->get_instance_list()->add_instance(instance);
        std::string pin_name = def_net->pin(i);
        pin_name = ieda::Str::trimEscape(pin_name);
        pin = instance->get_pin_by_term(pin_name);
        if (pin == nullptr) {
          std::cout << "Can not find Pin in Pin list ... pin name = " << def_net->pin(i) << std::endl;
        } else {
          net->add_instance_pin(pin);
          setPinNet(pin);
        }
      } else {
        std::cout << "Can not find instance in instance list ... instance name = " << io_name << std::endl;
      }
    }
  }

  IdbRegularWireList* wire_list = net->get_wire_list();
  for (int i = 0; i < def_net->numWires(); ++i) {
    defiWire* def_wire = def_net->wire(i);

    IdbRegularWire* wire = wire_list->add_wire(nullptr);
    wire->set_wire_state(def_wire->wireType());
    if (wire->get_wire_statement() == IdbWiringStatement::kShield) {
      wire->set_shield_name(def_wire->wireShieldNetName());
    }

    int32_t path_num = def_wire->numPaths();
    wire->init(path_num);
    for (int j = 0; j < path_num; ++j) {
      IdbRegularWireSegment* segment = wire->add_segment(nullptr);
      defiPath* def_path = def_wire->path(j);
      def_path->initTraverse();

      int32_t path_id = 0;
      while ((path_id = def_path->next()) != DEFIPATH_DONE) {
        switch (path_id) {
          case DEFIPATH_LAYER: {
            segment->set_layer_name(def_path->getLayer());
            segment->set_layer(layer_list->find_layer(def_path->getLayer()));
            break;
          }
          case DEFIPATH_VIA: {
            segment->set_is_via(true);

            IdbVias* via_list_def = design->get_via_list();
            IdbVias* via_list_lef = layout->get_via_list();
            IdbVia* via = via_list_def->find_via(def_path->getVia());
            if (via == nullptr) {
              via = via_list_lef->find_via(def_path->getVia());
            }

            if (via == nullptr) {
              std::cout << "Error : can not find the via = " << def_path->getVia() << std::endl;
              break;
            }

            IdbCoordinate<int32_t>* coordinate = segment->get_point_end();
            IdbVia* via_new = segment->copy_via(via);
            if (via_new != nullptr) {
              via_new->set_coordinate(coordinate);
            }

            break;
          }
          case DEFIPATH_VIAROTATION:
            break;
          case DEFIPATH_WIDTH:
            break;
          case DEFIPATH_POINT: {
            int x;
            int y;
            def_path->getPoint(&x, &y);
            segment->add_point(x, y);
            break;
          }

          case DEFIPATH_FLUSHPOINT: {
            int x;
            int y;
            int ext;
            def_path->getFlushPoint(&x, &y, &ext);
            //--------------tbd----------------
            segment->add_point(x, y);

            break;
          }
          case DEFIPATH_VIRTUALPOINT: {
            int x, y;
            def_path->getVirtualPoint(&x, &y);
            segment->add_virtual_point(x, y);
            break;
          }
          case DEFIPATH_SHAPE:
            break;
          case DEFIPATH_STYLE:
            break;
          case DEFIPATH_TAPERRULE:
            break;
          case DEFIPATH_VIADATA:
            break;
          case DEFIPATH_RECT: {
            int ll_x;
            int ll_y;
            int ur_x;
            int ur_y;
            def_path->getViaRect(&ll_x, &ll_y, &ur_x, &ur_y);
            segment->set_is_rect(true);
            segment->set_delta_rect(ll_x, ll_y, ur_x, ur_y);
            break;
          }
          case DEFIPATH_MASK:
            break;
          case DEFIPATH_VIAMASK:
            break;
          default:
            break;
        }
      }
    }
  }

  if (net_list->get_num() % 1000 == 0) {
    std::cout << "-" << std::flush;

    if (net_list->get_num() % 100000 == 0) {
      std::cout << std::endl;
    }
  }

  //   std::cout << "Parse net success... net name = " << net->get_net_name() << std::endl;

  return kDbSuccess;
}

int32_t DefRead::netEndCallback(defrCallbackType_e type, void*, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Net] ..." << std::endl;
    return kDbFail;
  }

  std::cout << std::endl;

  return kDbSuccess;
}

int32_t DefRead::specialNetBeginCallback(defrCallbackType_e type, int def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Special Net Number] ..." << std::endl;
    return kDbFail;
  }

  std::cout << "Begin parse Specialnet." << std::endl;

  return kDbSuccess;
}

int32_t DefRead::specialNetCallback(defrCallbackType_e type, defiNet* def_net, defiUserData data)
{
  if (def_net == nullptr) {
    std::cout << "Special Net is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Special Net] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_special_net(def_net);

  return kDbSuccess;
}

int32_t DefRead::parse_special_net(defiNet* def_net)
{
  if (def_net == nullptr) {
    std::cout << "Special Net is nullPtr..." << std::endl;
    return kDbFail;
  }

  if (def_net->hasUse()) {
    auto* enum_property = IdbEnum::GetInstance()->get_connect_property();
    if (enum_property->is_pdn(def_net->use())) {
      return parse_pdn(def_net);
    }

    if (enum_property->is_net(def_net->use())) {
      return parse_net(def_net);
    }
  }
  return kDbSuccess;
}

int32_t DefRead::parse_pdn(defiNet* def_net)
{
  IdbDesign* design = _def_service->get_design();  // Def
  // IdbLayout* layout = _def_service->get_layout();  // Lef
  // IdbLayers* layer_list = layout->get_layers();
  IdbPins* io_pin_list = design->get_io_pin_list();
  IdbInstanceList* instance_list = design->get_instance_list();

  IdbSpecialNetList* net_list = design->get_special_net_list();
  IdbSpecialNet* net = net_list->add_net(def_net->name());

  if (net == nullptr) {
    std::cout << "Create Net Error..." << std::endl;
    return kDbFail;
  }

  if (def_net->hasUse()) {
    net->set_connect_type(def_net->use());
  }

  if (def_net->hasSource()) {
    net->set_source_type(def_net->source());
  }

  if (def_net->hasWeight()) {
    net->set_weight(def_net->weight());
  }

  if (def_net->hasOriginal()) {
    net->set_original_net_name(def_net->original());
  }

  for (int i = 0; i < def_net->numConnections(); i++) {
    string io_name = def_net->instance(i);
    io_name = ieda::Str::trimEscape(io_name);
    IdbPin* pin = nullptr;
    if (io_name.compare("*") == 0) {
      net->add_pin_string(def_net->pin(i));
    } else if (io_name.compare("PIN") == 0) {
      std::string pin_name = def_net->pin(i);
      pin_name = ieda::Str::trimEscape(pin_name);
      pin = io_pin_list->find_pin(pin_name);
      if (pin == nullptr) {
        std::cout << "Can not find Pin in Pin list ... pin name = " << def_net->pin(i) << std::endl;
      } else {
        net->add_io_pin(pin);
        pin->set_special_net(net);
      }
    } else {
      IdbInstance* instance = instance_list->find_instance(io_name);
      if (instance != nullptr) {
        net->add_instance(instance);
        std::string pin_name = def_net->pin(i);
        pin_name = ieda::Str::trimEscape(pin_name);
        pin = instance->get_pin_by_term(pin_name);
        if (pin == nullptr) {
          std::cout << "Can not find Pin in Pin list ... pin name = " << def_net->pin(i) << std::endl;
        } else {
          net->add_instance_pin(pin);
          pin->set_special_net(net);
        }
      } else {
        std::cout << "Can not find instance in instance list ... instance name = " << io_name << std::endl;
      }
    }
  }

  vector<string> io_name_array = net->get_pin_string_list();
  if (io_name_array.size() > 0) {
    instance_list->get_pin_list_by_names(io_name_array, net->get_instance_pin_list(), net->get_instance_list());
  }

  IdbSpecialWireList* wire_list = net->get_wire_list();
  parse_pdn_wire(def_net, wire_list);
  parse_pdn_rects(def_net, wire_list);

  if (net_list->get_num() % 1000 == 0) {
    std::cout << "-" << std::flush;

    if (net_list->get_num() % 100000 == 0) {
      std::cout << std::endl;
    }
  }

  return kDbSuccess;
}

int32_t DefRead::parse_pdn_wire(defiNet* def_net, IdbSpecialWireList* wire_list)
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();

  for (int i = 0; i < def_net->numWires(); ++i) {
    defiWire* def_wire = def_net->wire(i);
    IdbSpecialWire* wire = wire_list->add_wire(nullptr);
    wire->set_wire_state(def_wire->wireType());
    if (wire->get_wire_state() == IdbWiringStatement::kShield) {
      wire->set_shield_name(def_wire->wireShieldNetName());
    }

    int32_t path_num = def_wire->numPaths();
    wire->init(path_num);
    for (int j = 0; j < path_num; ++j) {
      IdbSpecialWireSegment* segment = wire->add_segment(nullptr);
      defiPath* def_path = def_wire->path(j);
      def_path->initTraverse();

      int32_t path_id = 0;
      while ((path_id = def_path->next()) != DEFIPATH_DONE) {
        switch (path_id) {
          case DEFIPATH_LAYER: {
            segment->set_layer(layer_list->find_layer(def_path->getLayer()));
            break;
          }
          case DEFIPATH_VIA: {
            segment->set_is_via(true);

            IdbVias* via_list_def = design->get_via_list();
            IdbVias* via_list_lef = layout->get_via_list();
            IdbVia* via = via_list_def->find_via(def_path->getVia());
            if (via == nullptr) {
              via = via_list_lef->find_via(def_path->getVia());
            }

            //---------------tbd----------------
            IdbVia* via_new = segment->copy_via(via);
            if (via_new != nullptr) {
              via_new->set_coordinate(segment->get_point_start());
            }
            break;
          }
          case DEFIPATH_VIAROTATION:
            break;
          case DEFIPATH_WIDTH: {
            segment->set_route_width(def_path->getWidth());
            break;
          }
          case DEFIPATH_POINT: {
            int32_t x;
            int32_t y;
            def_path->getPoint(&x, &y);
            segment->add_point(x, y);
            break;
          }

          case DEFIPATH_FLUSHPOINT:
            break;
          case DEFIPATH_SHAPE: {
            segment->set_shape_type(def_path->getShape());
            break;
          }
          case DEFIPATH_STYLE: {
            segment->set_style(def_path->getStyle());
            break;
          }
          case DEFIPATH_TAPERRULE:
            break;
          case DEFIPATH_VIADATA:
            break;
          case DEFIPATH_RECT:
            break;
          case DEFIPATH_VIRTUALPOINT:
            break;
          case DEFIPATH_MASK:
            break;
          case DEFIPATH_VIAMASK:
            break;
          default:
            break;
        }
      }

      segment->set_bounding_box();
    }
  }
  return kDbSuccess;
}

int32_t DefRead::parse_pdn_rects(defiNet* def_net, IdbSpecialWireList* wire_list)
{
  // IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();

  for (int i = 0; i < def_net->numRectangles(); ++i) {
    IdbSpecialWire* wire = wire_list->add_wire(nullptr);
    IdbSpecialWireSegment* segment = wire->add_segment(nullptr);
    wire->set_wire_state(def_net->rectRouteStatus(i));
    if (wire->get_wire_state() == IdbWiringStatement::kShield) {
      wire->set_shield_name(def_net->rectRouteStatusShieldName(i));
    }

    /// set as rect to idb
    /// layer
    std::string layer = def_net->rectName(i);
    /// coordinate
    int llx = def_net->xl(i);
    int lly = def_net->yl(i);
    int urx = def_net->xh(i);
    int ury = def_net->yh(i);

    segment->set_shape_type(def_net->rectShapeType(i));
    segment->set_layer(layer_list->find_layer(layer));
    segment->set_is_rect(true);
    segment->set_delta_rect(llx, lly, urx, ury);

    segment->set_bounding_box();
  }

  return kDbSuccess;
}

int32_t DefRead::specialNetEndCallback(defrCallbackType_e type, void*, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Special Net] ..." << std::endl;
    return kDbFail;
  }

  std::cout << std::endl;

  std::cout << "End parse Specialnet." << std::endl;

  return kDbSuccess;
}

int32_t DefRead::pinsBeginCallback(defrCallbackType_e type, int def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Pin] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_pin_number(def_num);
  def_reader->set_start_time(clock());

  return kDbSuccess;
}

int32_t DefRead::parse_pin_number(int32_t def_pin_num)
{
  IdbDesign* design = _def_service->get_design();  // Def
  IdbPins* pin_list = design->get_io_pin_list();
  pin_list->init(def_pin_num);

  return kDbSuccess;
}

int32_t DefRead::pinCallback(defrCallbackType_e type, defiPin* def_pin, defiUserData data)
{
  if (def_pin == nullptr) {
    std::cout << "Pin is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Pin] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_pin(def_pin);

  return kDbSuccess;
}
/**
 * @brief Parse IO pins, create each IO Term in IdbPin
 *
 */
int32_t DefRead::parse_pin(defiPin* def_pin)
{
  if (def_pin == nullptr) {
    std::cout << "Pin is nullPtr..." << std::endl;
    return kDbFail;
  }

  IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();
  // IdbNetList* net_list = design->get_net_list();
  IdbPins* pin_list = design->get_io_pin_list();

  std::string pin_name = def_pin->pinName();
  std::string new_pin_name = ieda::Str::trimEscape(pin_name);

  IdbPin* pin = pin_list->add_pin_list(new_pin_name);
  if (pin == nullptr) {
    std::cout << "Create Pin Error..." << std::endl;
    return kDbFail;
  }

  std::string net_name = def_pin->netName();
  std::string new_net_name = ieda::Str::trimEscape(net_name);
  pin->set_net_name(new_net_name);
  // pin->set_net(net_list->find_net(pin->get_net_name()));
  pin->set_orient_by_enum(def_pin->orient());
  pin->set_as_io();
  // net ptr ----tbd---------

  IdbTerm* io_term = pin->set_term(nullptr);
  io_term->set_name(pin->get_pin_name());
  if (def_pin->hasDirection()) {
    io_term->set_direction(def_pin->direction());
  }

  if (def_pin->hasUse()) {
    io_term->set_type(def_pin->use());
  }

  if (def_pin->hasSpecial()) {
    io_term->set_special(true);
  }

  if (def_pin->numPorts() > 0) {
    io_term->set_has_port(true);
    for (int i = 0; i < def_pin->numPorts(); ++i) {
      IdbPort* port = io_term->add_port(nullptr);
      defiPinPort* def_port = def_pin->pinPort(i);
      port->set_orient_by_enum(def_port->orient());

      for (int i = 0; i < def_port->numLayer(); i++) {
        IdbLayerShape* shape = port->add_layer_shape();
        shape->set_type_rect();
        IdbLayer* layer = layer_list->find_layer(def_port->layer(i));
        shape->set_layer(layer);
        int32_t ll_x, ll_y, ur_x, ur_y;
        def_port->bounds(i, &ll_x, &ll_y, &ur_x, &ur_y);
        shape->add_rect(ll_x, ll_y, ur_x, ur_y);
      }

      if (def_port->hasPlacement()) {
        if (def_pin->use() != nullptr) {
          io_term->set_type(def_pin->use());
        }
        if (def_port->isPlaced()) {
          port->set_placement_status_place();
        } else if (def_port->isCover()) {
          port->set_placement_status_cover();
        } else if (def_port->isFixed()) {
          port->set_placement_status_fix();
        } else if (!def_port->hasPlacement()) {
          port->set_placement_status_unplace();
        } else {
        }

        port->set_coordinate(def_port->placementX(), def_port->placementY());

        if (i == 0) {
          if (def_pin->use() != nullptr) {
            io_term->set_type(def_pin->use());
          }
          if (def_port->isPlaced()) {
            io_term->set_placement_status_place();
          } else if (def_port->isCover()) {
            io_term->set_placement_status_cover();
          } else if (def_port->isFixed()) {
            io_term->set_placement_status_fix();
          } else if (!def_port->hasPlacement()) {
            io_term->set_placement_status_unplace();
          } else {
          }
        }
      }
    }

    pin->set_port_layer_shape();

  } else {
    int32_t bounding_box_ll_x = INT_MAX;
    int32_t bounding_box_ll_y = INT_MAX;
    int32_t bounding_box_ur_x = INT_MIN;
    int32_t bounding_box_ur_y = INT_MIN;

    io_term->set_has_port(false);

    // Support Layer
    if (def_pin->hasLayer()) {
      IdbPort* port = io_term->add_port(nullptr);

      int32_t layer_num = def_pin->numLayer();
      int32_t coordinate_x = 0;
      int32_t coordinate_y = 0;
      for (int i = 0; i < layer_num; i++) {
        IdbLayerShape* shape = port->add_layer_shape();
        shape->set_type_rect();
        IdbLayer* layer = layer_list->find_layer(def_pin->layer(i));
        shape->set_layer(layer);
        int32_t ll_x, ll_y, ur_x, ur_y;
        def_pin->bounds(i, &ll_x, &ll_y, &ur_x, &ur_y);
        shape->add_rect(ll_x, ll_y, ur_x, ur_y);

        bounding_box_ll_x = std::min(bounding_box_ll_x, ll_x);
        bounding_box_ll_y = std::min(bounding_box_ll_y, ll_y);
        bounding_box_ur_x = std::max(bounding_box_ur_x, ur_x);
        bounding_box_ur_y = std::max(bounding_box_ur_y, ur_y);

        // calculate average coodinate of all the ports
        int32_t mid_x = (ll_x + ur_x);
        int32_t mid_y = (ll_y + ur_y);

        coordinate_x = coordinate_x + mid_x;
        coordinate_y = coordinate_y + mid_y;
      }

      if (layer_num > 0) {
        io_term->set_average_position(coordinate_x / (layer_num * 2), coordinate_y / (layer_num * 2));
        io_term->set_bounding_box(bounding_box_ll_x, bounding_box_ll_y, bounding_box_ur_x, bounding_box_ur_y);
      } else {
        return kDbSuccess;
      }

      if (def_pin->hasPlacement()) {
        if (def_pin->use() != nullptr) {
          io_term->set_type(def_pin->use());
        }
        if (def_pin->isPlaced()) {
          io_term->set_placement_status_place();
        } else if (def_pin->isCover()) {
          io_term->set_placement_status_cover();
        } else if (def_pin->isFixed()) {
          io_term->set_placement_status_fix();
        } else if (def_pin->isUnplaced()) {
          io_term->set_placement_status_unplace();
        } else {
        }

        pin->set_location(def_pin->placementX(), def_pin->placementY());
        pin->set_average_coordinate(def_pin->placementX() + io_term->get_average_position().get_x(),
                                    def_pin->placementY() + io_term->get_average_position().get_y());
        pin->set_bounding_box();
      }
    }
  }

  // do not support polygon in v0.1
  //!<--------------tbd----------------
  // Polygon

  //   cout << "Parse Pin success... pin name = " << def_pin->pinName() << endl;

  return kDbSuccess;
}

int32_t DefRead::pinsEndCallback(defrCallbackType_e type, void*, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Pin] ..." << std::endl;
    return kDbFail;
  }

  return kDbSuccess;
}

int32_t DefRead::viaBeginCallback(defrCallbackType_e type, int def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Via] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_via_num(def_num);

  return kDbSuccess;
}

int32_t DefRead::parse_via_num(int32_t via_num)
{
  IdbDesign* design = _def_service->get_design();  // Def

  IdbVias* via_list = design->get_via_list();
  via_list->init_via_list(via_num);

  return kDbSuccess;
}

int32_t DefRead::viaCallback(defrCallbackType_e type, defiVia* def_via, defiUserData data)
{
  if (def_via == nullptr) {
    std::cout << "Via is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Via] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_via(def_via);

  return kDbSuccess;
}

int32_t DefRead::parse_via(defiVia* def_via)
{
  if (def_via == nullptr) {
    std::cout << "Via is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // Def
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbViaRuleList* _rule_list = layout->get_via_rule_list();
  IdbLayers* layer_list = layout->get_layers();

  IdbVias* via_list = design->get_via_list();
  IdbVia* via_instance = via_list->add_via(def_via->name());
  IdbViaMaster* master_instance = via_instance->get_instance();

  if (def_via->hasViaRule()) {
    IdbViaMasterGenerate* master_generate = master_instance->get_master_generate();
    master_instance->set_type_generate();

    char* rule_name;
    int32_t cutsize_x, cutsize_y;
    char *layer_bottom_name, *layer_cut_name, *layer_top_name;
    int32_t cut_spacing_x, cut_spacing_y;
    int32_t enclosure_bottom_x, enclosure_bottom_y, enclosure_top_x, enclosure_top_y;

    def_via->viaRule(&rule_name, &cutsize_x, &cutsize_y, &layer_bottom_name, &layer_cut_name, &layer_top_name, &cut_spacing_x,
                     &cut_spacing_y, &enclosure_bottom_x, &enclosure_bottom_y, &enclosure_top_x, &enclosure_top_y);

    IdbViaRuleGenerate* via_rule = _rule_list->find_via_rule_generate(rule_name);
    master_generate->set_rule_name(rule_name);
    master_generate->set_rule_generate(via_rule);
    master_generate->set_cut_size(cutsize_x, cutsize_y);
    IdbLayer* layer_bottom = layer_list->find_layer(layer_bottom_name);
    master_generate->set_layer_bottom(dynamic_cast<IdbLayerRouting*>(layer_bottom));
    IdbLayerCut* layer_cut = dynamic_cast<IdbLayerCut*>(layer_list->find_layer(layer_cut_name));
    layer_cut->set_via_rule(via_rule);
    master_generate->set_layer_cut(layer_cut);
    IdbLayer* layer_top = layer_list->find_layer(layer_top_name);
    master_generate->set_layer_top(dynamic_cast<IdbLayerRouting*>(layer_top));
    master_generate->set_cut_spacing(cut_spacing_x, cut_spacing_y);
    master_generate->set_enclosure_bottom(enclosure_bottom_x, enclosure_bottom_y);
    master_generate->set_enclosure_top(enclosure_top_x, enclosure_top_y);

    int32_t original_offset_x = 0;
    int32_t original_offset_y = 0;
    if (def_via->hasOrigin()) {
      def_via->origin(&original_offset_x, &original_offset_y);
      master_generate->set_original(original_offset_x, original_offset_y);
    }

    if (def_via->hasOffset()) {
      int32_t offset_bottom_x, offset_bottom_y, offset_top_x, offset_top_y;
      def_via->offset(&offset_bottom_x, &offset_bottom_y, &offset_top_x, &offset_top_y);
      master_generate->set_offset_bottom(offset_bottom_x, offset_bottom_y);
      master_generate->set_offset_top(offset_top_x, offset_top_y);
    }

    // ROWCOL numCutRows numCutCols
    // Specifies the number of cut rows and columns that make up the cut array.
    // Default: 1, for both values
    // Type: Positive integer, for both values
    // --lefdef_reference_5.8 p817
    {
      int32_t num_rows = 1;
      int32_t num_cols = 1;
      if (def_via->hasRowCol()) {
        def_via->rowCol(&num_rows, &num_cols);
      }
      master_generate->set_cut_row_col(num_rows, num_cols);
      master_instance->set_cut_row_col(num_rows, num_cols);

      /// if pattern exist, cut array must follow the pattern rule
      if (def_via->hasCutPattern()) {
        master_generate->set_patttern(def_via->cutPattern());
      }

      // build core cut shape
      vector<IdbRect*> cut_rect_list = master_generate->get_cut_rect_list();

      int32_t cut_width_total = num_cols * cutsize_x + (num_cols - 1) * cut_spacing_x;
      int32_t cut_height_total = num_rows * cutsize_y + (num_rows - 1) * cut_spacing_y;

      int32_t ll_x_min = (-cut_width_total / 2) + original_offset_x;
      int32_t ll_y_min = (-cut_height_total / 2) + original_offset_y;
      for (int32_t i = 0; i < num_rows; ++i) {
        for (int32_t j = 0; j < num_cols; j++) {
          /// if pattern exist, cut shape must o
          if (nullptr != master_generate->get_patttern() && !master_generate->is_pattern_cut_exist(i, j)) {
            continue;
          }
          int32_t ll_x = ll_x_min + j * (cutsize_x + cut_spacing_x);
          int32_t ll_y = ll_y_min + i * (cutsize_y + cut_spacing_y);
          int32_t ur_x = ll_x + cutsize_x;
          int32_t ur_y = ll_y + cutsize_y;
          master_generate->add_cut_rect(ll_x, ll_y, ur_x, ur_y);
        }
      }

      master_generate->set_cut_bouding_rect(ll_x_min, ll_y_min, ll_x_min + cut_width_total, ll_y_min + cut_height_total);

      master_instance->set_via_shape();
    }
  } else {
    master_instance->set_type_fixed();
    // Fixed via
    int32_t min_x = INT_MAX;
    int32_t min_y = INT_MAX;
    int32_t max_x = INT_MIN;
    int32_t max_y = INT_MIN;
    for (int i = 0; i < def_via->numLayers(); ++i) {
      char* layer_name;
      int32_t ll_x, ll_y, ur_x, ur_y;
      def_via->layer(i, &layer_name, &ll_x, &ll_y, &ur_x, &ur_y);
      IdbViaMasterFixed* master_fixed = master_instance->add_fixed(layer_name);
      IdbLayer* layer = layer_list->find_layer(layer_name);
      if (layer == nullptr) {
        return kDbFail;
      }

      master_fixed->set_layer(layer);
      master_fixed->add_rect(ll_x, ll_y, ur_x, ur_y);

      // record the core area of cut
      if (layer->get_type() == IdbLayerType::kLayerCut) {
        min_x = std::min(min_x, ll_x);
        min_y = std::min(min_y, ll_y);
        max_x = std::max(max_x, ur_x);
        max_y = std::max(max_y, ur_y);
      }
    }

    {
      int32_t num_rows = 1;
      int32_t num_cols = 1;
      if (def_via->hasRowCol()) {
        def_via->rowCol(&num_rows, &num_cols);
      }
      master_instance->set_cut_row_col(num_rows, num_cols);
    }

    master_instance->set_cut_rect(min_x, min_y, max_x, max_y);
    master_instance->set_via_shape();
  }

  return kDbSuccess;
}

int32_t DefRead::blockageCallback(defrCallbackType_e type, defiBlockage* def_blockage, defiUserData data)
{
  if (def_blockage == nullptr) {
    std::cout << "Blockage is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Blockage] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_blockage(def_blockage);

  return kDbSuccess;
}

int32_t DefRead::parse_blockage(defiBlockage* def_blockage)
{
  if (def_blockage == nullptr) {
    std::cout << "Blockage is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // Def
  IdbBlockageList* blockage_list = design->get_blockage_list();
  IdbInstanceList* instance_list = design->get_instance_list();
  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbLayers* layer_list = layout->get_layers();

  if (def_blockage->hasLayer()) {
    // routing blockage
    IdbRoutingBlockage* routing_blockage = blockage_list->add_blockage_routing(def_blockage->layerName());
    routing_blockage->set_layer(layer_list->find_layer(def_blockage->layerName()));

    if (def_blockage->hasSlots()) {
      routing_blockage->set_slots(true);
    }

    if (def_blockage->hasFills()) {
      routing_blockage->set_fills(true);
    }

    if (def_blockage->hasPushdown()) {
      routing_blockage->set_pushdown(true);
    }

    if (def_blockage->hasExceptpgnet()) {
      routing_blockage->set_except_pgnet(true);
    }

    if (def_blockage->hasComponent()) {
      routing_blockage->set_instance_name(def_blockage->layerComponentName());
      IdbInstance* instance = instance_list->find_instance(def_blockage->layerComponentName());
      routing_blockage->set_instance(instance);
    }

    if (def_blockage->hasSpacing()) {
      routing_blockage->set_min_spacing(def_blockage->minSpacing());
    }

    if (def_blockage->hasDesignRuleWidth()) {
      routing_blockage->set_effective_width(def_blockage->designRuleWidth());
    }

    for (int i = 0; i < def_blockage->numRectangles(); ++i) {
      routing_blockage->add_rect(def_blockage->xl(i), def_blockage->yl(i), def_blockage->xh(i), def_blockage->yh(i));
    }

    // do not support polygon
    //-----------------tbd----------------------
  } else {
    // placement blockage
    IdbPlacementBlockage* placement_blockage = blockage_list->add_blockage_placement();

    if (def_blockage->hasSoft()) {
      placement_blockage->set_soft(true);
    }

    if (def_blockage->hasPartial()) {
      placement_blockage->set_max_density(def_blockage->placementMaxDensity());
    }

    if (def_blockage->hasComponent()) {
      placement_blockage->set_instance_name(def_blockage->layerComponentName());
      IdbInstance* instance = instance_list->find_instance(def_blockage->layerComponentName());
      placement_blockage->set_instance(instance);
    }

    for (int i = 0; i < def_blockage->numRectangles(); ++i) {
      placement_blockage->add_rect(def_blockage->xl(i), def_blockage->yl(i), def_blockage->xh(i), def_blockage->yh(i));
    }
  }

  return kDbSuccess;
}

int32_t DefRead::gcellGridCallback(defrCallbackType_e type, defiGcellGrid* def_grid, defiUserData data)
{
  if (def_grid == nullptr) {
    std::cout << "GCell Grid is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : GCell Grid] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_gcell_grid(def_grid);

  return kDbSuccess;
}

int32_t DefRead::parse_gcell_grid(defiGcellGrid* def_grid)
{
  if (def_grid == nullptr) {
    std::cout << "GCell Grid is nullPtr..." << std::endl;
    return kDbFail;
  }

  IdbLayout* layout = _def_service->get_layout();  // Lef
  IdbGCellGridList* gcell_grid_list = layout->get_gcell_grid_list();
  IdbGCellGrid* gcell_grid = gcell_grid_list->add_gcell_grid(nullptr);
  if (def_grid->macro()[0] == 'X') {
    gcell_grid->set_direction(IdbTrackDirection::kDirectionX);
  } else {
    gcell_grid->set_direction(IdbTrackDirection::kDirectionY);
  }

  gcell_grid->set_num(def_grid->xNum());
  gcell_grid->set_start(def_grid->x());
  gcell_grid->set_space(def_grid->xStep());

  return kDbSuccess;
}

int32_t DefRead::regionCallback(defrCallbackType_e type, defiRegion* def_region, defiUserData data)
{
  if (def_region == nullptr) {
    std::cout << "Region is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Region] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_region(def_region);

  return kDbSuccess;
}

int32_t DefRead::parse_region(defiRegion* def_region)
{
  if (def_region == nullptr) {
    std::cout << "Region is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // def
  IdbRegionList* region_list = design->get_region_list();
  IdbRegion* region = region_list->add_region(def_region->name());

  if (def_region->hasType()) {
    region->set_type(def_region->type());
  }

  for (int i = 0; i < def_region->numRectangles(); ++i) {
    region->add_boundary(def_region->xl(i), def_region->yl(i), def_region->xh(i), def_region->yh(i));
  }

  // Property
  //---------------tbd-------------------------

  return kDbSuccess;
}

int32_t DefRead::slotsCallback(defrCallbackType_e type, defiSlot* def_slot, defiUserData data)
{
  if (def_slot == nullptr) {
    std::cout << "Slot is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Slot] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_slot(def_slot);

  return kDbSuccess;
}

int32_t DefRead::parse_slot(defiSlot* def_slot)
{
  if (def_slot == nullptr) {
    std::cout << "Slot is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // def
  IdbSlotList* slot_list = design->get_slot_list();
  IdbSlot* slot = slot_list->add_slot();

  if (def_slot->hasLayer()) {
    slot->set_layer_name(def_slot->layerName());
    for (int i = 0; i < def_slot->numRectangles(); ++i) {
      slot->add_rect(def_slot->xl(i), def_slot->yl(i), def_slot->xh(i), def_slot->yh(i));
    }
  }

  // Polygon
  //---------------tbd-------------------------

  return kDbSuccess;
}

int32_t DefRead::groupCallback(defrCallbackType_e type, defiGroup* def_group, defiUserData data)
{
  if (def_group == nullptr) {
    std::cout << "Group is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Group] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_group(def_group);

  return kDbSuccess;
}

int32_t DefRead::parse_group(defiGroup* def_group)
{
  if (def_group == nullptr) {
    std::cout << "Group is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // def
  IdbRegionList* region_list = design->get_region_list();
  // IdbInstanceList* instance_list = design->get_instance_list();
  IdbGroupList* group_list = design->get_group_list();

  IdbGroup* group = group_list->add_group(def_group->name());
  group->set_region(region_list->find_region(def_group->regionName()));

  // compNamePattern
  // Property
  //---------------tbd------------------

  return kDbSuccess;
}

int32_t DefRead::fillsCallback(defrCallbackType_e type, int32_t def_num, defiUserData data)
{
  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Fill] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_fill_number(def_num);
  def_reader->set_start_time(clock());

  return kDbSuccess;
}

int32_t DefRead::parse_fill_number(int32_t def_fill_num)
{
  // IdbDesign* design = _def_service->get_design();  // Def

  return kDbSuccess;
}

int32_t DefRead::fillCallback(defrCallbackType_e type, defiFill* def_fill, defiUserData data)
{
  if (def_fill == nullptr) {
    std::cout << "Fill is nullPtr..." << std::endl;
    return kDbFail;
  }

  DefRead* def_reader = (DefRead*) data;
  if (!def_reader->check_type(type)) {
    std::cout << "Check Type Error [Def : Fill] ..." << std::endl;
    return kDbFail;
  }

  def_reader->parse_fill(def_fill);

  return kDbSuccess;
}

int32_t DefRead::parse_fill(defiFill* def_fill)
{
  if (def_fill == nullptr) {
    std::cout << "Fill is nullPtr..." << std::endl;
    return kDbFail;
  }
  IdbDesign* design = _def_service->get_design();  // def
  IdbLayout* layout = _def_service->get_layout();  // lef
  IdbLayers* layer_list = layout->get_layers();
  IdbFillList* fill_list = design->get_fill_list();

  if (def_fill->hasLayer()) {
    IdbLayer* layer = layer_list->find_layer(def_fill->layerName());
    IdbFillLayer* fill_layer = fill_list->add_fill_layer(layer);
    if (fill_layer) {
      for (int i = 0; i < def_fill->numRectangles(); ++i) {
        fill_layer->add_rect(def_fill->xl(i), def_fill->yl(i), def_fill->xh(i), def_fill->yh(i));
      }

      // Polygon
      //------------------tbd----------------------------
    }
  }

  if (def_fill->hasVia()) {
    IdbVias* via_list_def = design->get_via_list();
    IdbVias* via_list_lef = layout->get_via_list();
    IdbVia* via = via_list_def->find_via(def_fill->viaName());
    if (via == nullptr) {
      via = via_list_lef->find_via(def_fill->viaName());
    }
    IdbVia* via_new = via->clone();
    IdbFillVia* fill_via = fill_list->add_fill_via(via_new);
    if (via_new != nullptr) {
      for (int i = 0; i < def_fill->numViaPts(); ++i) {
        defiPoints def_point_list = def_fill->getViaPts(i);
        for (int j = 0; j < def_point_list.numPoints; ++j) {
          fill_via->add_coordinate(def_point_list.x[j], def_point_list.y[j]);
        }
      }
    }
  }

  return kDbSuccess;
}

int32_t DefRead::busBitCharsCallBack(defrCallbackType_e c, const char* bus_bit_chars_str, defiUserData data)
{
  if (c != defrBusBitCbkType) {
    std::cout << "busBitCharsCB callback type unmatch!" << std::endl;
    return kDbFail;
  }
  if (bus_bit_chars_str == nullptr) {
    std::cout << "BusBitChars is nullptr..." << std::endl;
    return kDbFail;
  }
  if (strlen(bus_bit_chars_str) != 2) {
    std::cout << "Unsupported Bus Bit Chars..." << std::endl;
    return kDbFail;
  }

  auto* def_reader = static_cast<DefRead*>(data);
  if (!def_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : BusBitChars] ..." << std::endl;
    return kDbFail;
  }
  int32_t parse_status = def_reader->parse_bus_bit_chars(bus_bit_chars_str);

  return parse_status;
}

int32_t DefRead::parse_bus_bit_chars(const char* bus_bit_chars_str)
{
  IdbDesign* design = this->get_service()->get_design();
  IdbBusBitChars* bus_bit_chars = new IdbBusBitChars();
  bus_bit_chars->setLeftDelimiter(bus_bit_chars_str[0]);
  bus_bit_chars->setRightDelimter(bus_bit_chars_str[1]);

  if (design->get_bus_bit_chars() != nullptr) {
    delete design->get_bus_bit_chars();
  }
  design->set_bus_bit_chars(bus_bit_chars);
  return kDbSuccess;
}

}  // namespace idb
