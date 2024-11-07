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

#include "lm_layout_init.h"

namespace ilm {

void LmLayoutInit::initProcessNode()
{
  initLayerIds();
  initViaIds();
  initCellMasters();
}

void LmLayoutInit::initDesign()
{
  initDie();
  initCore();
  initTracks();
  initPDN();
  initInstances();
  initNets();
}

void LmLayoutInit::initLayerIds()
{
}

void LmLayoutInit::initViaIds()
{
}

void LmLayoutInit::initCellMasters()
{
}

void LmLayoutInit::initDie()
{
}

void LmLayoutInit::initCore()
{
}

void LmLayoutInit::initTracks()
{
}

void LmLayoutInit::initPDN()
{
}

void LmLayoutInit::initInstances()
{
}

void LmLayoutInit::initNets()
{
}

}  // namespace ilm