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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-08 10:20:28
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 16:03:41
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/AbacusLegalizer.hh
 * @Description: Legalization with abacus method
 *
 *
 */
#ifndef IPL_ABACUSLEGALIZER_H
#define IPL_ABACUSLEGALIZER_H

#include "Config.hh"
#include "config/LegalizerConfig.hh"
#include "database/LGDatabase.hh"

namespace ipl {

#define AbacusLegalizerInst (ipl::AbacusLegalizer::getInst())

enum class LG_MODE
{
  kNone,
  kComplete,
  kIncremental
};

class AbacusLegalizer
{
 public:
  static AbacusLegalizer& getInst();
  static void destoryInst();
  void initAbacusLegalizer(Config* pl_config, PlacerDB* placer_db);

  void updateInstanceList();
  void updateInstanceList(std::vector<Instance*> inst_list);

  LG_MODE get_mode() const { return _mode; }
  bool runLegalize();
  bool runIncrLegalize();

  bool isInitialized() { return _mode != LG_MODE::kNone; }

 private:
  static AbacusLegalizer* _abacus_lg_instance;

  LGConfig _config;
  LGDatabase _database;
  int32_t _row_height;
  int32_t _site_width;

  LG_MODE _mode;
  std::vector<LGInstance*> _target_inst_list;

  AbacusLegalizer() = default;
  AbacusLegalizer(const AbacusLegalizer&) = delete;
  AbacusLegalizer(AbacusLegalizer&&) = delete;
  ~AbacusLegalizer();
  AbacusLegalizer& operator=(const AbacusLegalizer&) = delete;
  AbacusLegalizer& operator=(AbacusLegalizer&&) = delete;

  void initLGConfig(Config* pl_config);
  void initLGDatabase(PlacerDB* placer_db);
  void initLGLayout();
  void wrapRowList();
  void wrapRegionList();
  void wrapCellList();
  void initSegmentList();

  bool checkMapping();
  LGInstance* findLGInstance(Instance* pl_inst);
  bool checkInstChanged(Instance* pl_inst, LGInstance* lg_inst);
  void updateInstanceInfo(Instance* pl_inst, LGInstance* lg_inst);
  void updateInstanceMapping(Instance* pl_inst, LGInstance* lg_inst);

  bool runCompleteMode();
  bool runIncrementalMode();

  void pickAndSortMovableInstList(std::vector<LGInstance*>& movable_inst_list);
  int32_t placeRow(LGInstance* inst, int32_t row_idx, bool is_trial);
  int32_t searchNearestIntervalIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape);
  int32_t searchRemainSpaceSegIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape, int32_t origin_index);
  LGCluster arrangeInstIntoIntervalCluster(LGInstance* inst, LGInterval* interval);
  void replaceClusterInfo(LGCluster& cluster);
  void arrangeClusterMinXCoordi(LGCluster& cluster);
  void legalizeCluster(LGCluster& cluster);
  int32_t obtainFrontMaxX(LGCluster& cluster);
  int32_t obtainBackMinX(LGCluster& cluster);

  int32_t calDistanceWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x);
  bool checkOverlapWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x);

  void alignInstanceOrient(std::vector<LGInstance*> inst_list);
  void writebackPlacerDB(std::vector<LGInstance*> inst_list);
  int32_t calTotalMovement();
};

}  // namespace ipl

#endif