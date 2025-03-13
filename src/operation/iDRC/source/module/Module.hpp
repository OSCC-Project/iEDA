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

namespace idrc {

#define DRCMOD (idrc::Module::getInst())

class Module
{
 public:
  static void initInst();
  static Module& getInst();
  static void destroyInst();
  // function
  void check();

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
  void checkAdjacentCutSpacing();
  void checkCornerFillSpacing();
  void checkCutEOLSpacing();
  void checkCutShort();
  void checkDifferentLayerCutSpacing();
  void checkEnclosure();
  void checkEnclosureEdge();
  void checkEnclosureParallel();
  void checkEndOfLineSpacing();
  void checkFloatingPatch();
  void checkJogToJogSpacing();
  void checkMaxViaStack();
  void checkMetalShort();
  void checkMinHole();
  void checkMinimumArea();
  void checkMinimumCut();
  void checkMinimumWidth();
  void checkMinStep();
  void checkNonsufficientMetalOverlap();
  void checkNotchSpacing();
  void checkOffGridOrWrongWay();
  void checkOutOfDie();
  void checkParallelRunLengthSpacing();
  void checkSameLayerCutSpacing();
};

}  // namespace idrc
