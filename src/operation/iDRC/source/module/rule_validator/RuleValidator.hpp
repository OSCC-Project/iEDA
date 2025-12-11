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
#include "RVCluster.hpp"
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
  std::vector<Violation> verify(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list,
                                std::set<ViolationType>& drc_check_type_set, std::vector<DRCShape>& drc_check_region_list);

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
  RVModel initRVModel(std::vector<DRCShape>& drc_env_shape_list, std::vector<DRCShape>& drc_result_shape_list, std::set<ViolationType>& drc_check_type_set,
                      std::vector<DRCShape>& drc_check_region_list);
  void setRVComParam(RVModel& rv_model);
  void buildRVClusterList(RVModel& rv_model);
  void verifyRVModel(RVModel& rv_model);
  void buildRVCluster(RVCluster& rv_cluster);
  bool needVerifying(RVCluster& rv_cluster);
  void buildEnvViolation(RVCluster& rv_cluster);
  void verifyRVCluster(RVCluster& rv_cluster);
  void verifyAdjacentCutSpacing(RVCluster& rv_cluster);
  void verifyCornerFillSpacing(RVCluster& rv_cluster);
  void verifyCornerSpacing(RVCluster& rv_cluster);
  void verifyCutEOLSpacing(RVCluster& rv_cluster);
  void verifyCutShort(RVCluster& rv_cluster);
  void verifyDifferentLayerCutSpacing(RVCluster& rv_cluster);
  void verifyEnclosure(RVCluster& rv_cluster);
  void verifyEnclosureEdge(RVCluster& rv_cluster);
  void verifyEnclosureParallel(RVCluster& rv_cluster);
  void verifyEndOfLineSpacing(RVCluster& rv_cluster);
  void verifyFloatingPatch(RVCluster& rv_cluster);
  void verifyJogToJogSpacing(RVCluster& rv_cluster);
  void verifyMaximumWidth(RVCluster& rv_cluster);
  void verifyMaxViaStack(RVCluster& rv_cluster);
  void verifyMetalShort(RVCluster& rv_cluster);
  void verifyMinHole(RVCluster& rv_cluster);
  void verifyMinimumArea(RVCluster& rv_cluster);
  void verifyMinimumCut(RVCluster& rv_cluster);
  void verifyMinimumWidth(RVCluster& rv_cluster);
  void verifyMinStep(RVCluster& rv_cluster);
  void verifyNonsufficientMetalOverlap(RVCluster& rv_cluster);
  void verifyNotchSpacing(RVCluster& rv_cluster);
  void verifyOffGridOrWrongWay(RVCluster& rv_cluster);
  void verifyOutOfDie(RVCluster& rv_cluster);
  void verifyParallelRunLengthSpacing(RVCluster& rv_cluster);
  void verifySameLayerCutSpacing(RVCluster& rv_cluster);
  bool needVerifying(RVCluster& rv_cluster, ViolationType violation_type);
  void processRVCluster(RVCluster& rv_cluster);
  void buildViolationList(RVCluster& rv_cluster);
  void buildViolationList(RVModel& rv_model);

#if 1  // aux
  int32_t getIdx(int32_t idx, int32_t coord_size);
#endif

#if 1  // debug
  void debugPlotRVModel(RVModel& rv_model, std::string flag);
  void debugPlotRVCluster(RVCluster& rv_cluster, std::string flag);
  void debugOutputViolation(RVModel& rv_model);
#endif
};

}  // namespace idrc
