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
#include "GPDataType.hpp"
#include "Logger.hpp"
#include "RVBox.hpp"
#include "RVModel.hpp"

namespace idrc {

#define DRCRV (idrc::RuleValidator::getInst())

class RuleValidator
{
 public:
  static void initInst();
  static RuleValidator& getInst();
  static void destroyInst();
  // function
  std::vector<Violation> verify(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list,const std::string option = "");

 private:
  // self
  static RuleValidator* _rv_instance;

  RuleValidator() = default;
  RuleValidator(const RuleValidator& other) = delete;
  RuleValidator(RuleValidator&& other) = delete;
  ~RuleValidator() = default;
  RuleValidator& operator=(const RuleValidator& other) = delete;
  RuleValidator& operator=(RuleValidator&& other) = delete;
  // function
  RVModel initRVModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list);
  void setRVComParam(RVModel& rv_model);
  void buildRVBoxList(RVModel& rv_model);
  void verifyRVModel(RVModel& rv_model);
  bool needVerifying(RVBox& rv_box);
  void buildViolationSet(RVBox& rv_box);
  void verifyRVBox(RVBox& rv_box);
  void verifyAdjacentCutSpacing(RVBox& rv_box);
  void verifyCornerFillSpacing(RVBox& rv_box);
  void verifyCornerSpacing(RVBox& rv_box);
  void verifyCutEOLSpacing(RVBox& rv_box);
  void verifyCutShort(RVBox& rv_box);
  void verifyDifferentLayerCutSpacing(RVBox& rv_box);
  void verifyEnclosure(RVBox& rv_box);
  void verifyEnclosureEdge(RVBox& rv_box);
  void verifyEnclosureParallel(RVBox& rv_box);
  void verifyEndOfLineSpacing(RVBox& rv_box);
  void verifyFloatingPatch(RVBox& rv_box);
  void verifyJogToJogSpacing(RVBox& rv_box);
  void verifyMaximumWidth(RVBox& rv_box);
  void verifyMaxViaStack(RVBox& rv_box);
  void verifyMetalShort(RVBox& rv_box);
  void verifyMinHole(RVBox& rv_box);
  void verifyMinimumArea(RVBox& rv_box);
  void verifyMinimumCut(RVBox& rv_box);
  void verifyMinimumWidth(RVBox& rv_box);
  void verifyMinStep(RVBox& rv_box);
  void verifyNonsufficientMetalOverlap(RVBox& rv_box);
  void verifyNotchSpacing(RVBox& rv_box);
  void verifyOffGridOrWrongWay(RVBox& rv_box);
  void verifyOutOfDie(RVBox& rv_box);
  void verifyParallelRunLengthSpacing(RVBox& rv_box);
  void verifySameLayerCutSpacing(RVBox& rv_box);
  void processRVBox(RVBox& rv_box);
  void buildViolationList(RVBox& rv_box);
  void updateSummary(RVBox& rv_box);
  void buildViolationList(RVModel& rv_model);
  void updateSummary(RVModel& rv_model);
  void printSummary(RVModel& rv_model);

#if 1  // aux
  int32_t getIdx(int32_t idx, int32_t coord_size);
#endif

#if 1  // debug
  void debugPlotRVModel(RVModel& rv_model, std::string flag);
  void debugPlotRVBox(RVBox& rv_box, std::string flag);
  void debugViolationByType(RVBox& rv_box, ViolationType violation_type);
  void debugVerifyRVModelByGolden(RVModel& rv_model);
  void debugVerifyRVBoxByGolden(RVBox& rv_box);
  void debugOutputViolationByGolden(RVBox& rv_box);
#endif
};

}  // namespace idrc
