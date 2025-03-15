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

#include "DRCBox.hpp"
#include "DRCModel.hpp"
#include "DataManager.hpp"
#include "Logger.hpp"

namespace idrc {

#define DRCMOD (idrc::Module::getInst())

class Module
{
 public:
  static void initInst();
  static Module& getInst();
  static void destroyInst();
  // function
  void check(DRCModel& drc_model);

 private:
  // self
  static Module* _mod_instance;

  Module() = default;
  Module(const Module& other) = delete;
  Module(Module&& other) = delete;
  ~Module() = default;
  Module& operator=(const Module& other) = delete;
  Module& operator=(Module&& other) = delete;
  // function
  void buildDRCModel(DRCModel& drc_model);
  void checkDRCModel(DRCModel& drc_model);

  void checkDRCBox(DRCBox& drc_box);
  void checkAdjacentCutSpacing(DRCBox& drc_box);
  void checkCornerFillSpacing(DRCBox& drc_box);
  void checkCutEOLSpacing(DRCBox& drc_box);
  void checkCutShort(DRCBox& drc_box);
  void checkDifferentLayerCutSpacing(DRCBox& drc_box);
  void checkEnclosure(DRCBox& drc_box);
  void checkEnclosureEdge(DRCBox& drc_box);
  void checkEnclosureParallel(DRCBox& drc_box);
  void checkEndOfLineSpacing(DRCBox& drc_box);
  void checkFloatingPatch(DRCBox& drc_box);
  void checkJogToJogSpacing(DRCBox& drc_box);
  void checkMaxViaStack(DRCBox& drc_box);
  void checkMetalShort(DRCBox& drc_box);
  void checkMinHole(DRCBox& drc_box);
  void checkMinimumArea(DRCBox& drc_box);
  void checkMinimumCut(DRCBox& drc_box);
  void checkMinimumWidth(DRCBox& drc_box);
  void checkMinStep(DRCBox& drc_box);
  void checkNonsufficientMetalOverlap(DRCBox& drc_box);
  void checkNotchSpacing(DRCBox& drc_box);
  void checkOffGridOrWrongWay(DRCBox& drc_box);
  void checkOutOfDie(DRCBox& drc_box);
  void checkParallelRunLengthSpacing(DRCBox& drc_box);
  void checkSameLayerCutSpacing(DRCBox& drc_box);
};

}  // namespace idrc
