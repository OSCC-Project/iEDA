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

#include <stdint.h>

#include "engine_init.h"
#include "DRCViolationType.h"

/**
 * check geometry overlap method
 */

namespace idb {
class IdbLayerShape;
class IdbPin;
class IdbVia;
class IdbRect;
class IdbNet;
template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace idrc {

class DrcEngineManager;
enum class LayoutType;

class DrcEngineInitDef : DrcEngineInit
{
 public:
  DrcEngineInitDef(DrcEngineManager* engine_manager) : DrcEngineInit(engine_manager) {}
  ~DrcEngineInitDef() {}

  void init() override;

 private:
  void initDataFromIOPins();
  void initDataFromInstances();
  void initDataFromPDN();
  void initDataFromNets();
};

}  // namespace idrc