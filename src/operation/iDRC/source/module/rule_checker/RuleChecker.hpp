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

#include "DataManager.hpp"
#include "Logger.hpp"
#include "RCBox.hpp"
#include "RCModel.hpp"

namespace idrc {

#define DRCRC (idrc::RuleChecker::getInst())

class RuleChecker
{
 public:
  static void initInst();
  static RuleChecker& getInst();
  static void destroyInst();
  // function
  std::vector<Violation> check(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list);

 private:
  // self
  static RuleChecker* _rc_instance;

  RuleChecker() = default;
  RuleChecker(const RuleChecker& other) = delete;
  RuleChecker(RuleChecker&& other) = delete;
  ~RuleChecker() = default;
  RuleChecker& operator=(const RuleChecker& other) = delete;
  RuleChecker& operator=(RuleChecker&& other) = delete;
  // function
  RCModel initRCModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list);
  void buildRCModel(RCModel& rc_model);
  void checkRCModel(RCModel& rc_model);
  void checkRCBox(RCBox& rc_box);
  void checkAdjacentCutSpacing(RCBox& rc_box);
  void checkCornerFillSpacing(RCBox& rc_box);
  void checkCutEOLSpacing(RCBox& rc_box);
  void checkCutShort(RCBox& rc_box);
  void checkDifferentLayerCutSpacing(RCBox& rc_box);
  void checkEnclosure(RCBox& rc_box);
  void checkEnclosureEdge(RCBox& rc_box);
  void checkEnclosureParallel(RCBox& rc_box);
  void checkEndOfLineSpacing(RCBox& rc_box);
  void checkFloatingPatch(RCBox& rc_box);
  void checkJogToJogSpacing(RCBox& rc_box);
  void checkMaxViaStack(RCBox& rc_box);
  void checkMetalShort(RCBox& rc_box);
  void checkMinHole(RCBox& rc_box);
  void checkMinimumArea(RCBox& rc_box);
  void checkMinimumCut(RCBox& rc_box);
  void checkMinimumWidth(RCBox& rc_box);
  void checkMinStep(RCBox& rc_box);
  void checkNonsufficientMetalOverlap(RCBox& rc_box);
  void checkNotchSpacing(RCBox& rc_box);
  void checkOffGridOrWrongWay(RCBox& rc_box);
  void checkOutOfDie(RCBox& rc_box);
  void checkParallelRunLengthSpacing(RCBox& rc_box);
  void checkSameLayerCutSpacing(RCBox& rc_box);
  std::vector<Violation> getViolationList(RCModel& rc_model);
};

}  // namespace idrc
