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

#ifndef IPL_POST_GLOBAL_PLACER_H
#define IPL_POST_GLOBAL_PLACER_H

#include "Config.hh"
#include "PlacerDB.hh"
#include "config/PostGPConfig.hh"
#include "database/PostGPDatabase.hh"
#include "timing/TimingAnnotation.hh"

namespace ipl {

class PostGP
{
 public:
  PostGP() = delete;
  PostGP(Config* pl_config, PlacerDB* placer_db);
  PostGP(const PostGP&) = delete;
  PostGP(PostGP&&) = delete;
  ~PostGP();

  PostGP& operator=(const PostGP&) = delete;
  PostGP& operator=(PostGP&&) = delete;

  void runBufferBalancing();
  void runCellBalancing();
  void runLoadReduction();

  void runInstRelocationForTimingOpt();

 private:
  PostGPConfig _config;
  PostGPDatabase _database;

  // timing opt
  TimingAnnotation* _timing_annotation;

  bool doBufferBalancing(Instance* buffer);

  void updateTargetLogicInstances();
  void updateTargetFlipflopInstances();

  Point<float> calEarlyLateCost(Instance* inst);
  std::pair<Point<int32_t>, Point<int32_t>> buildStaticSearchWindow(Instance* inst);
  std::pair<Point<int32_t>, Point<int32_t>> buildDynamicSearchWindow(Instance* inst);
  std::vector<Point<int32_t>> obtainCandidateLocations(Instance* inst, Point<int32_t> anchor_point);

  bool runIncrementalLegalization(std::vector<Instance*> target_inst_list);  // Mode: NEAREST, EXACT
  bool rollBackToBestLocation(Instance* inst);
};

}  // namespace ipl

#endif