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

#include <map>
#include <stack>
#include <unordered_map>

#include "AbacusCluster.hh"
#include "LGMethodInterface.hh"

namespace ipl {
class LGInstance;
class LGInterval;
template <class T>
class Rectangle;
}  // namespace ipl

namespace ieda_solver {

struct RollbackInfo
{
  RollbackInfo() : is_dirty(false) {}
  ~RollbackInfo() = default;

  bool is_dirty;
  // for incremental legalization rollback
  std::vector<AbacusCluster> origin_clusters;
  std::vector<AbacusCluster> addition_clusters;
};

class Abacus : public LGMethodInterface
{
 public:
  Abacus() = default;
  Abacus(const Abacus&) = delete;
  Abacus(Abacus&&) = delete;
  ~Abacus();

  Abacus& operator=(const Abacus&) = delete;
  Abacus& operator=(Abacus&&) = delete;

  void initDataRequirement(ipl::LGConfig* lg_config, ipl::LGDatabase* lg_database) override;
  bool isInitialized() override;
  void specifyTargetInstList(std::vector<ipl::LGInstance*>& target_inst_list) override;
  bool runLegalization() override;
  bool runIncrLegalization() override;
  bool runRollback(bool clear_but_not_rollback) override;

 private:
  std::unordered_map<std::string, AbacusCluster*> _cluster_map;
  std::vector<AbacusCluster*> _inst_belong_cluster;
  std::vector<AbacusCluster*> _interval_cluster_root;
  std::vector<int32_t> _interval_remain_length;

  int32_t _row_height = -1;
  int32_t _site_width = -1;

  std::stack<RollbackInfo> _rollback_stack;

  void pickAndSortMovableInstList(std::vector<ipl::LGInstance*>& movable_inst_list);
  int32_t placeRow(ipl::LGInstance* inst, int32_t row_idx, bool is_trial, bool is_record_cluster);
  int32_t searchNearestIntervalIndex(std::vector<ipl::LGInterval*>& segment_list, ipl::Rectangle<int32_t>& inst_shape);
  int32_t searchRemainSpaceSegIndex(std::vector<ipl::LGInterval*>& segment_list, ipl::Rectangle<int32_t>& inst_shape, int32_t origin_index);
  AbacusCluster arrangeInstIntoIntervalCluster(ipl::LGInstance* inst, ipl::LGInterval* interval);
  void replaceClusterInfo(AbacusCluster& cluster, bool is_record_cluster);
  void arrangeClusterMinXCoordi(AbacusCluster& cluster);
  void legalizeCluster(AbacusCluster& cluster);
  void mergeWithPreviousCluster(AbacusCluster& cluster, AbacusCluster prev_cluster);
  void mergeWithNextCluster(AbacusCluster& cluster, AbacusCluster next_cluster);
  int32_t obtainFrontMaxX(AbacusCluster& cluster);
  int32_t obtainBackMinX(AbacusCluster& cluster);

  int32_t calDistanceWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x);
  bool checkOverlapWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x);

  AbacusCluster* findCluster(std::string cluster_name);
  void insertCluster(std::string name, AbacusCluster* cluster);
  void deleteCluster(std::string name);

  void updateRemainLength(ipl::LGInterval* interval, int32_t delta);
  void splitTargetInst(ipl::LGInstance* inst, RollbackInfo& rollback_info);
  void deleteTargetIntervalClusters(ipl::LGInterval* interval, std::vector<AbacusCluster>& cluster_list);
  void insertTargetIntervalClusters(ipl::LGInterval* interval, std::vector<AbacusCluster>& cluster_list);
  void insertClusterChainIntoInterval(ipl::LGInterval* interval, std::vector<AbacusCluster*>& cluster_chain);
  void reCalIntervalRemainLength(ipl::LGInterval* interval);
  std::string obtainUniqueClusterName(std::string origin_name);

  void debugIntervalRemainLength(std::string interval_name);

  int get_best_cost_row(ipl::LGInstance* inst);
};

}  // namespace ieda_solver
