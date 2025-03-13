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
#include "Module.hpp"

namespace idrc {

// public

void Module::initInst()
{
  if (_mod_instance == nullptr) {
    _mod_instance = new Module();
  }
}

Module& Module::getInst()
{
  if (_mod_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_mod_instance;
}

void Module::destroyInst()
{
  if (_mod_instance != nullptr) {
    delete _mod_instance;
    _mod_instance = nullptr;
  }
}

// function

void Module::check()
{
  checkAdjacentCutSpacing();
  checkCornerFillSpacing();
  checkCutEOLSpacing();
  checkCutShort();
  checkDifferentLayerCutSpacing();
  checkEnclosure();
  checkEnclosureEdge();
  checkEnclosureParallel();
  checkEndOfLineSpacing();
  checkFloatingPatch();
  checkJogToJogSpacing();
  checkMaxViaStack();
  checkMetalShort();
  checkMinHole();
  checkMinimumArea();
  checkMinimumCut();
  checkMinimumWidth();
  checkMinStep();
  checkNonsufficientMetalOverlap();
  checkNotchSpacing();
  checkOffGridOrWrongWay();
  checkOutOfDie();
  checkParallelRunLengthSpacing();
  checkSameLayerCutSpacing();
}

// private

Module* Module::_mod_instance = nullptr;

}  // namespace idrc
