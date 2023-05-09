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
#include "InstanceSwap.hh"

#include "utility/Utility.hh"

namespace ipl {

InstanceSwap::InstanceSwap(DPConfig* config, DPDatabase* database, DPOperator* dp_operator)
{
  _config = config;
  _database = database;
  _operator = dp_operator;
  _row_height = database->get_layout()->get_row_height();
  _site_width = database->get_layout()->get_site_width();
}

InstanceSwap::~InstanceSwap()
{
}

void InstanceSwap::runGlobalSwap()
{
  bool is_clusted = _operator->checkIfClustered();
  if (!is_clusted) {
    _operator->updateInstClustering();
  }

  // step 1: sort inst based on their hpwl benefit
  std::vector<DPInstance*> movable_inst_list;
  sortInstBasedHPWLBenefit(movable_inst_list);

  int64_t total_benefit = 0;

  for (auto* inst : movable_inst_list) {
    Rectangle<int32_t> optimal_region = _operator->obtainOptimalCoordiRegion(inst);

    // step 2: select optimal region candidate
    std::vector<std::pair<Point<int32_t>, DPInstance*>> candidate_list;
    searchCandidateCoordiList(optimal_region, inst, candidate_list);

    // step 3: trially place or swap, calculate the benefit
    std::pair<Point<int32_t>, DPInstance*> best_candidate;
    int64_t best_benefit = 0;
    for (auto pair : candidate_list) {
      int64_t swap_benefit = 0;

      if (!pair.second) {
        swap_benefit = placeInstance(inst, pair.first.get_x(), pair.first.get_y(), true);
      } else {
        swap_benefit = swapInstance(inst, pair.second, true);
      }

      if (swap_benefit > 0) {  // record swap candidate when has benefit for accelerating
        best_benefit = swap_benefit;
        best_candidate = pair;
        break;
      }
    }

    // skip when no benefit
    if (best_benefit <= 0) {
      continue;
    }

    // step 4: place or swap decision
    if (!best_candidate.second) {
      placeInstance(inst, best_candidate.first.get_x(), best_candidate.first.get_y(), false);
    } else {
      swapInstance(inst, best_candidate.second, false);
    }

    total_benefit += best_benefit;
  }
  // LOG_INFO << "Expected Total HPWL Benefit: " << total_benefit;
}

void InstanceSwap::runVerticalSwap()
{
  bool is_clusted = _operator->checkIfClustered();
  if (!is_clusted) {
    _operator->updateInstClustering();
  }

  int64_t total_benefit = 0;
  for (auto* inst : _database->get_design()->get_inst_list()) {
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }

    // step 1: select optimal row candidate
    std::vector<std::pair<Point<int32_t>, DPInstance*>> candidate_list;
    std::pair<int32_t, int32_t> optimal_line = _operator->obtainOptimalYCoordiLine(inst);
    searchImproveYCoordiList(optimal_line, inst, 1, candidate_list);

    // step 2: trially place or swap, calculate the benefit
    std::pair<Point<int32_t>, DPInstance*> best_candidate;
    int64_t best_benefit = 0;

    for (auto pair : candidate_list) {
      int64_t swap_benefit = 0;

      if (!pair.second) {
        swap_benefit = placeInstance(inst, pair.first.get_x(), pair.first.get_y(), true);
      } else {
        swap_benefit = swapInstance(inst, pair.second, true);
      }

      if (swap_benefit > 0) {  // record swap candidate when has benefit for accelerating
        best_benefit = swap_benefit;
        best_candidate = pair;
        break;
      }
    }

    // skip when no benefit
    if (best_benefit <= 0) {
      continue;
    }

    // step 4: place or swap decision
    if (!best_candidate.second) {
      placeInstance(inst, best_candidate.first.get_x(), best_candidate.first.get_y(), false);
    } else {
      swapInstance(inst, best_candidate.second, false);
    }

    // LOG_INFO << "Expected HPWL Benefit: " << best_benefit;
    total_benefit += best_benefit;
  }
  // LOG_INFO << "Expected Total HPWL Benefit: " << total_benefit;
}

void InstanceSwap::sortInstBasedHPWLBenefit(std::vector<DPInstance*>& movable_inst_list)
{
  std::map<int64_t, std::vector<DPInstance*>, std::greater<int64_t>> inst_map;

  for (auto* inst : _database->get_design()->get_inst_list()) {
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }

    Rectangle<int32_t> optimal_region = std::move(_operator->obtainOptimalCoordiRegion(inst));
    int64_t benefit = placeInstance(inst, optimal_region.get_ll_x(), optimal_region.get_ll_y(), true);

    auto it = inst_map.find(benefit);
    if (it != inst_map.end()) {
      it->second.push_back(inst);
    } else {
      std::vector<DPInstance*> inst_list{inst};
      inst_map.emplace(benefit, inst_list);
    }
  }

  for (auto pair : inst_map) {
    for (auto* inst : pair.second) {
      movable_inst_list.push_back(inst);
    }
  }
}

void InstanceSwap::searchCandidateCoordiList(Rectangle<int32_t>& optimal_region, DPInstance* inst,
                                             std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list)
{
  Utility utility;
  int32_t origin_x = inst->get_coordi().get_x();
  int32_t origin_y = inst->get_coordi().get_y();
  // aready in optimal region
  if ((origin_x >= optimal_region.get_ll_x() && origin_x <= optimal_region.get_ur_x())
      && (origin_y >= optimal_region.get_ll_y() && origin_y <= optimal_region.get_ur_y())) {
    return;
  }

  int32_t inst_width = inst->get_shape().get_width();
  std::pair<int32_t, int32_t> row_range = utility.obtainMinMaxIdx(0, _row_height, optimal_region.get_ll_y(), optimal_region.get_ur_y());

  bool case1_flag = false;  // optimal x in front of all intervals
  bool case2_flag = false;  // optimal x between intervals
  bool case3_flag = false;  // optimal x behind all intervals

  auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();
  for (int32_t i = row_range.first; i < row_range.second; i++) {
    auto& interval_list = interval_2d_list[i];
    if (interval_list.empty()) {
      continue;
    }

    auto* first_interval = *(interval_list.begin());
    auto* last_interval = interval_list.back();
    if (optimal_region.get_ur_x() < first_interval->get_min_x()) {
      case1_flag = true;
    }
    if (optimal_region.get_ll_x() > last_interval->get_max_x()) {
      case3_flag = true;
    }

    if (case1_flag) {
      int32_t min_x = first_interval->get_min_x();
      optimal_region.set_rectangle(min_x, optimal_region.get_ll_y(), min_x, optimal_region.get_ur_y());
      case1_flag = false;
      continue;
    }
    if (case3_flag) {
      int32_t max_x = last_interval->get_max_x() - inst_width;
      optimal_region.set_rectangle(max_x, optimal_region.get_ll_y(), max_x, optimal_region.get_ur_y());
      case3_flag = false;
      continue;
    }

    DPInterval* front_interval = first_interval;
    for (auto* interval : interval_2d_list[i]) {
      if (optimal_region.get_ur_x() < interval->get_min_x() && optimal_region.get_ll_x() > front_interval->get_max_x()) {
        case2_flag = true;
      }
      if (case2_flag) {
        int32_t min_x = front_interval->get_max_x() - inst_width;
        int32_t max_x = interval->get_min_x();
        optimal_region.set_rectangle(min_x, optimal_region.get_ll_y(), max_x, optimal_region.get_ur_y());
        fillIntervalCandidateList(front_interval, optimal_region.get_ll_x(), optimal_region.get_ur_x(), inst_width, candidate_list);
        fillIntervalCandidateList(interval, optimal_region.get_ll_x(), optimal_region.get_ur_x(), inst_width, candidate_list);
        case2_flag = false;
        break;
      }

      bool overlap_flag
          = _operator->checkOverlap(interval->get_min_x(), interval->get_max_x(), optimal_region.get_ll_x(), optimal_region.get_ur_x());
      if (overlap_flag) {
        fillIntervalCandidateList(interval, optimal_region.get_ll_x(), optimal_region.get_ur_x(), inst_width, candidate_list);
      }

      front_interval = interval;

      // Debug.
      if (candidate_list.size() > 5) {
        return;
      }
    }
  }
}

void InstanceSwap::searchImproveYCoordiList(std::pair<int32_t, int32_t>& optimal_line, DPInstance* inst, int32_t row_range,
                                            std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list)
{
  int32_t origin_y = inst->get_coordi().get_y();
  // already in optimal line
  if (origin_y >= optimal_line.first && origin_y <= optimal_line.second) {
    return;
  }

  int32_t inst_width = inst->get_shape().get_width();
  int32_t inst_min_x = inst->get_coordi().get_x();
  int32_t inst_max_x = inst->get_shape().get_ur_x();
  int32_t row_index = origin_y / _row_height;
  auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();

  if (origin_y < optimal_line.first) {
    for (auto* interval : interval_2d_list[row_index + 1]) {
      fillIntervalCandidateList(interval, inst_min_x, inst_max_x, inst_width, candidate_list);
    }
  }

  if (origin_y > optimal_line.second) {
    for (auto* interval : interval_2d_list[row_index - 1]) {
      fillIntervalCandidateList(interval, inst_min_x, inst_max_x, inst_width, candidate_list);
    }
  }
}

void InstanceSwap::fillIntervalCandidateList(DPInterval* interval, int32_t query_min, int32_t query_max, int32_t inst_width,
                                             std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list)
{
  std::pair<int32_t, int32_t> overlap_range
      = _operator->obtainOverlapRange(interval->get_min_x(), interval->get_max_x(), query_min, query_max);
  int32_t coordi_y = interval->get_belong_row()->get_coordinate().get_y();
  int32_t coordi_x = overlap_range.first;
  int32_t max_x = overlap_range.second;

  auto* cur_cluster = interval->get_cluster_root();

  if (!cur_cluster) {
    return;
  }

  auto* last_cluster = cur_cluster;
  while (cur_cluster) {
    int32_t cluster_min_x = cur_cluster->get_min_x();
    int32_t cluster_max_x = cur_cluster->get_max_x();

    // white space before cluster_min_x
    if (coordi_x < cluster_min_x) {
      if (cluster_min_x - coordi_x >= inst_width) {
        candidate_list.push_back(std::make_pair(Point<int32_t>(coordi_x, coordi_y), nullptr));
      }
      coordi_x = cluster_min_x;
    }

    if (coordi_x > max_x) {
      break;
    }

    // overlap with cluster
    int32_t inst_x = cluster_min_x;
    if (coordi_x < cluster_max_x && max_x >= cluster_min_x) {
      for (auto* inst : cur_cluster->get_inst_list()) {
        if (coordi_x > max_x) {
          break;
        }

        int32_t inst_width = inst->get_shape().get_width();
        if (_operator->checkOverlap(coordi_x, max_x, inst_x, inst_x + inst_width)) {
          candidate_list.push_back(std::make_pair(Point<int32_t>(inst_x, coordi_y), inst));
          coordi_x = inst_x + inst_width;
        }
        inst_x += inst_width;
      }
      if (max_x <= cluster_max_x) {
        break;
      } else {
        coordi_x = cluster_max_x;
      }
    }

    auto* back_cluster = cur_cluster->get_back_cluster();
    if (back_cluster) {
      last_cluster = back_cluster;  // record last cluster
    }
    cur_cluster = back_cluster;
  }
  if (max_x >= last_cluster->get_max_x()) {
    if (interval->get_max_x() - coordi_x >= inst_width) {
      candidate_list.push_back(std::make_pair(Point<int32_t>(coordi_x, coordi_y), nullptr));
    }
  }
}

int64_t InstanceSwap::placeInstance(DPInstance* inst, int32_t x_coordi, int32_t y_coordi, bool is_trial)
{
  int32_t origin_x = inst->get_coordi().get_x();
  int32_t origin_y = inst->get_coordi().get_y();
  auto origin_orient = inst->get_orient();

  int64_t origin_hpwl = _operator->calInstAffectiveHPWL(inst);

  // TODO: search the orient
  Rectangle<int32_t> trial_inst_shape(x_coordi, y_coordi, x_coordi + inst->get_shape().get_width(),
                                      y_coordi + inst->get_shape().get_height());
  auto* cur_interval = obtainCorrespondingInterval(trial_inst_shape);
  DPRow* cur_row = nullptr;
  if (cur_interval) {
    cur_row = cur_interval->get_belong_row();
  } else {
    int32_t row_index = inst->get_coordi().get_y() / _row_height;
    cur_row = _database->get_layout()->get_row_2d_list()[row_index][0];
  }

  inst->set_orient(cur_row->get_row_orient());
  inst->updateCoordi((x_coordi / _site_width) * _site_width, (y_coordi / _row_height) * _row_height);

  int64_t modify_hpwl = _operator->calInstAffectiveHPWL(inst);

  if (is_trial) {
    inst->set_orient(origin_orient);
    inst->updateCoordi(origin_x, origin_y);

    // tmp fixed bug
    if (cur_interval) {
      if (cur_interval->get_remain_length() - inst->get_shape().get_width() < 0) {
        return INT64_MIN;
      }
    }
  } else {
    eraseInstAndSplitCluster(inst->get_belong_cluster(), inst);

    // cur_interval must exist
    if (!cur_interval) {
      LOG_ERROR << "Instance: " << inst->get_name() << " new position is not in any interval";
      return INT64_MIN;
    }
    updateAloneInstToInterval(inst, cur_interval);
  }
  return (origin_hpwl - modify_hpwl);
}

int64_t InstanceSwap::swapInstance(DPInstance* inst_1, DPInstance* inst_2, bool is_trial)
{
  DPCluster* cluster_1 = inst_1->get_belong_cluster();
  DPCluster* cluster_2 = inst_2->get_belong_cluster();
  DPInterval* interval_1 = cluster_1->get_belong_interval();
  DPInterval* interval_2 = cluster_2->get_belong_interval();
  int32_t delta_width = inst_1->get_shape().get_width() - inst_2->get_shape().get_width();
  int32_t interval1_remain = interval_1->get_remain_length() + delta_width;
  int32_t interval2_remain = interval_2->get_remain_length() - delta_width;
  if (interval1_remain < 0 || interval1_remain > interval_1->get_max_length()) {
    return INT64_MIN;
  }
  if (interval2_remain < 0 || interval2_remain > interval_2->get_max_length()) {
    return INT64_MIN;
  }

  Point<int32_t> inst1_coordi = std::move(inst_1->get_coordi());
  Point<int32_t> inst2_coordi = std::move(inst_2->get_coordi());
  auto inst1_orient = inst_1->get_orient();
  auto inst2_orient = inst_2->get_orient();
  int32_t inst1_internal_id = inst_1->get_internal_id();
  int32_t inst2_internal_id = inst_2->get_internal_id();

  int64_t origin_hpwl = _operator->calInstPairAffectiveHPWL(inst_1, inst_2);

  if (cluster_1 == cluster_2) {
    cluster_1->replaceInstance(inst_1, inst2_internal_id);
    cluster_1->replaceInstance(inst_2, inst1_internal_id);
    std::vector<DPInstance*> mark_insts{inst_1, inst_2};
    updateClusterInstCoordi(*cluster_1, mark_insts, is_trial);
    int64_t modify_hpwl = _operator->calInstPairAffectiveHPWL(inst_1, inst_2);
    int64_t sum_movement = 0;
    if (is_trial) {
      sum_movement = calOtherInstMovement(*cluster_1, mark_insts);
      // recover insts coordinates
      inst_1->updateCoordi(inst1_coordi.get_x(), inst1_coordi.get_y());
      inst_2->updateCoordi(inst2_coordi.get_x(), inst2_coordi.get_y());
      // recover the order of cluster
      cluster_1->replaceInstance(inst_1, inst1_internal_id);
      cluster_1->replaceInstance(inst_2, inst2_internal_id);

    } else {
      inst_1->set_internal_id(inst2_internal_id);
      inst_2->set_internal_id(inst1_internal_id);
    }
    return (origin_hpwl - modify_hpwl) - sum_movement;
  }

  DPCluster tmp_cluster1 = *cluster_1;
  DPCluster tmp_cluster2 = *cluster_2;
  tmp_cluster1.replaceInstance(inst_2, inst1_internal_id);
  tmp_cluster2.replaceInstance(inst_1, inst2_internal_id);
  instantLegalizeCluster(tmp_cluster1);
  instantLegalizeCluster(tmp_cluster2);

  int64_t sum_movement = 0;
  inst_2->set_orient(interval_1->get_belong_row()->get_row_orient());
  inst_1->set_orient(interval_2->get_belong_row()->get_row_orient());
  std::vector<DPInstance*> mark1_list{inst_1};
  std::vector<DPInstance*> mark2_list{inst_2};
  updateClusterInstCoordi(tmp_cluster1, mark2_list, is_trial);
  updateClusterInstCoordi(tmp_cluster2, mark1_list, is_trial);

  int64_t modify_hpwl = _operator->calInstPairAffectiveHPWL(inst_1, inst_2);

  if (is_trial) {
    sum_movement += calOtherInstMovement(tmp_cluster1, mark2_list);
    sum_movement += calOtherInstMovement(tmp_cluster2, mark1_list);
    inst_1->set_orient(inst1_orient);
    inst_2->set_orient(inst2_orient);
    inst_1->updateCoordi(inst1_coordi.get_x(), inst1_coordi.get_y());
    inst_2->updateCoordi(inst2_coordi.get_x(), inst2_coordi.get_y());

    // avoid fusion of two cluster
    if (checkIfTwoClusterFusion1(tmp_cluster1, inst_1, inst_2) || checkIfTwoClusterFusion1(tmp_cluster2, inst_1, inst_2)) {
      origin_hpwl = INT64_MIN;
      modify_hpwl = 0;
      sum_movement = 0;
    }
    if (checkIfTwoClusterFusion2(tmp_cluster1, tmp_cluster2)) {
      origin_hpwl = INT64_MIN;
      modify_hpwl = 0;
      sum_movement = 0;
    }

  } else {
    inst_1->set_belong_cluster(cluster_2);
    inst_2->set_belong_cluster(cluster_1);
    inst_1->set_internal_id(inst2_internal_id);
    inst_2->set_internal_id(inst1_internal_id);
    replaceCluster(*cluster_1, tmp_cluster1);
    replaceCluster(*cluster_2, tmp_cluster2);

    // update interval remain length
    int32_t delta_width = inst_1->get_shape().get_width() - inst_2->get_shape().get_width();
    cluster_1->get_belong_interval()->updateRemainLength(delta_width);
    cluster_2->get_belong_interval()->updateRemainLength(-delta_width);
  }

  return (origin_hpwl - modify_hpwl) - sum_movement;
}

void InstanceSwap::updateAloneInstToInterval(DPInstance* inst, DPInterval* interval)
{
  if (!interval) {
    LOG_WARNING << "Cannot find the interval for current inst position";
    return;
  }

  int32_t inst_min_x = inst->get_coordi().get_x();
  int32_t inst_max_x = inst->get_shape().get_ur_x();
  int32_t inst_width = inst->get_shape().get_width();
  int32_t interval_min_x = interval->get_min_x();
  int32_t interval_max_x = interval->get_max_x();
  interval->updateRemainLength(-inst_width);

  DPCluster* new_cluster = nullptr;
  std::string cluster_name = inst->get_name() + "SWAP_" + interval->get_name() + "_" + std::to_string(inst_min_x);
  auto* exist_cluster = _database->get_design()->find_cluster(cluster_name);
  if (exist_cluster) {
    cluster_name = cluster_name + "_plus";
  }
  new_cluster = new DPCluster(cluster_name);

  new_cluster->set_min_x(inst_min_x);
  new_cluster->add_inst(inst);
  new_cluster->set_belong_interval(interval);
  new_cluster->set_boundary_min_x(interval_min_x);
  new_cluster->set_boundary_max_x(interval_max_x);
  inst->set_belong_cluster(new_cluster);
  inst->set_internal_id(0);

  auto* cur_cluster = interval->get_cluster_root();
  if (inst_min_x >= interval_min_x && inst_max_x <= cur_cluster->get_min_x()) {
    // set to the root
    interval->set_cluster_root(new_cluster);
    new_cluster->set_back_cluster(cur_cluster);
    cur_cluster->set_front_cluster(new_cluster);
  } else {
    auto* last_cluster = cur_cluster;
    while (cur_cluster) {
      int32_t cluster_max_x = cur_cluster->get_max_x();

      auto* back_cluster = cur_cluster->get_back_cluster();
      if (back_cluster) {
        last_cluster = back_cluster;
        if (inst_min_x >= cluster_max_x && inst_max_x <= back_cluster->get_min_x()) {
          new_cluster->set_front_cluster(cur_cluster);
          new_cluster->set_back_cluster(back_cluster);
          cur_cluster->set_back_cluster(new_cluster);
          back_cluster->set_front_cluster(new_cluster);
        }
      }
      cur_cluster = back_cluster;
    }
    if (inst_min_x >= last_cluster->get_max_x() && inst_max_x <= interval_max_x) {
      new_cluster->set_front_cluster(last_cluster);
      last_cluster->set_back_cluster(new_cluster);
    }
  }
  _database->get_design()->add_cluster(new_cluster);
}

DPInterval* InstanceSwap::obtainCurrentInterval(DPInstance* inst)
{
  DPInterval* target_interval = nullptr;
  int32_t inst_min_x = inst->get_coordi().get_x();
  int32_t inst_max_x = inst->get_shape().get_ur_x();
  int32_t row_index = inst->get_coordi().get_y() / _row_height;
  auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();
  for (auto* interval : interval_2d_list[row_index]) {
    if (interval->checkInLine(inst_min_x, inst_max_x)) {
      target_interval = interval;
      break;
    }
  }
  return target_interval;
}

DPInterval* InstanceSwap::obtainCorrespondingInterval(Rectangle<int32_t>& inst_shape)
{
  DPInterval* target_interval = nullptr;
  int32_t inst_min_x = inst_shape.get_ll_x();
  int32_t inst_max_x = inst_shape.get_ur_x();
  int32_t row_index = inst_shape.get_ll_y() / _row_height;
  auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();
  for (auto* interval : interval_2d_list[row_index]) {
    if (interval->checkInLine(inst_min_x, inst_max_x)) {
      target_interval = interval;
      break;
    }
  }
  return target_interval;
}

void InstanceSwap::instantLegalizeCluster(DPCluster& cluster)
{
  arrangeClusterMinXCoordi(cluster);
  int32_t cur_min_x, front_max_x, back_min_x;
  cur_min_x = cluster.get_min_x();
  front_max_x = obtainFrontMaxX(cluster);
  while (cur_min_x < front_max_x) {
    DPCluster front_cluster = *(cluster.get_front_cluster());
    temporarySpliceCluster(front_cluster, cluster);
    // tmp fix bug.
    front_cluster.set_name(cluster.get_name());
    arrangeClusterMinXCoordi(front_cluster);
    cur_min_x = front_cluster.get_min_x();
    front_max_x = obtainFrontMaxX(front_cluster);
    cluster = front_cluster;
  }

  cur_min_x = cluster.get_min_x();
  back_min_x = obtainBackMinX(cluster);
  while (cur_min_x + cluster.get_total_width() > back_min_x) {
    auto* back_cluster = cluster.get_back_cluster();
    if (!back_cluster) {
      //
    }
    DPCluster back_cluster_ref = *back_cluster;
    temporarySpliceCluster(cluster, back_cluster_ref);
    arrangeClusterMinXCoordi(cluster);
    cur_min_x = cluster.get_min_x();
    back_min_x = obtainBackMinX(cluster);
  }
}

void InstanceSwap::arrangeClusterMinXCoordi(DPCluster& cluster_ref)
{
  double weight_e = 0.0;
  double weight_q = 0.0;
  double total_width = 0;

  int32_t coordi_x = cluster_ref.get_min_x();
  for (auto* inst : cluster_ref.get_inst_list()) {
    weight_e += inst->get_weight();
    weight_q += inst->get_weight() * (coordi_x - total_width);
    int32_t inst_width = inst->get_shape().get_width();
    total_width += inst_width;
    coordi_x += inst_width;
  }

  int32_t cluster_x = weight_q / weight_e;

  cluster_x = (cluster_x / _site_width) * _site_width;
  cluster_x < cluster_ref.get_boundary_min_x() ? cluster_x = cluster_ref.get_boundary_min_x() : cluster_x;
  cluster_x + total_width > cluster_ref.get_boundary_max_x() ? cluster_x = cluster_ref.get_boundary_max_x() - total_width : cluster_x;

  cluster_ref.set_min_x(cluster_x);
}

int32_t InstanceSwap::obtainFrontMaxX(DPCluster& cluster)
{
  int32_t front_max_x = cluster.get_boundary_min_x();
  if (cluster.get_front_cluster()) {
    front_max_x = cluster.get_front_cluster()->get_max_x();
  }
  return front_max_x;
}

int32_t InstanceSwap::obtainBackMinX(DPCluster& cluster)
{
  int32_t back_min_x = cluster.get_boundary_max_x();
  if (cluster.get_back_cluster()) {
    back_min_x = cluster.get_back_cluster()->get_min_x();
  }
  return back_min_x;
}

void InstanceSwap::updateClusterInstCoordi(DPCluster& cluster, std::vector<DPInstance*>& special_insts, bool is_trial)
{
  int32_t y_coordi = cluster.get_belong_interval()->get_belong_row()->get_coordinate().get_y();
  int32_t x_coordi = cluster.get_min_x();
  for (auto* inst : cluster.get_inst_list()) {
    bool skip_flag = false;
    for (auto* special_inst : special_insts) {
      if (special_inst == inst) {
        special_inst->updateCoordi(x_coordi, y_coordi);
        skip_flag = true;
        break;
      }
    }

    if (!skip_flag && !is_trial) {
      inst->updateCoordi(x_coordi, y_coordi);
    }
    x_coordi += inst->get_shape().get_width();
  }
}

int64_t InstanceSwap::calOtherInstMovement(DPCluster& cluster, std::vector<DPInstance*>& except_insts)
{
  int64_t sum_movement = 0;
  int32_t x_coordi = cluster.get_min_x();
  for (auto* inst : cluster.get_inst_list()) {
    bool skip_flag = false;
    for (auto* except_inst : except_insts) {
      if (except_inst == inst) {
        skip_flag = true;
        break;
      }
    }

    if (!skip_flag) {
      sum_movement += (std::abs(x_coordi - inst->get_coordi().get_x()) * inst->get_weight() * 2);
    }

    x_coordi += inst->get_shape().get_width();
  }
  return sum_movement;
}

void InstanceSwap::replaceCluster(DPCluster& origin_cluster, DPCluster& modify_cluster)
{
  auto* origin_interval = origin_cluster.get_belong_interval();
  auto* origin_root = origin_interval->get_cluster_root();

  // may be collapsing with front or back cluster
  DPCluster* front_origin = origin_cluster.get_front_cluster();
  DPCluster* front_modify = modify_cluster.get_front_cluster();
  while (front_origin != front_modify) {
    if (front_origin == origin_root) {
      origin_interval->set_cluster_root(&origin_cluster);
    }

    std::string delete_cluster = front_origin->get_name();

    // next
    front_origin = front_origin->get_front_cluster();
    if (front_origin) {
      front_origin->set_back_cluster(&origin_cluster);
    }
    _database->get_design()->deleteCluster(delete_cluster);
  }

  DPCluster* back_origin = origin_cluster.get_back_cluster();
  DPCluster* back_modify = modify_cluster.get_back_cluster();
  while (back_origin != back_modify) {
    std::string delete_cluster = back_origin->get_name();

    // next
    back_origin = back_origin->get_back_cluster();
    if (back_origin) {
      back_origin->set_front_cluster(&origin_cluster);
    }
    _database->get_design()->deleteCluster(delete_cluster);
  }

  origin_cluster = std::move(modify_cluster);
  auto& inst_list = origin_cluster.get_inst_list();
  for (size_t i = 0; i < inst_list.size(); i++) {
    inst_list[i]->set_internal_id(i);
    inst_list[i]->set_belong_cluster(&origin_cluster);
  }
}

void InstanceSwap::temporarySpliceCluster(DPCluster& dest_cluster, DPCluster& src_cluster)
{
  const auto& inst_list = src_cluster.get_inst_list();
  for (auto* inst : inst_list) {
    dest_cluster.add_inst(inst);
  }
  dest_cluster.set_back_cluster(src_cluster.get_back_cluster());
}

bool InstanceSwap::checkIfTwoClusterFusion1(DPCluster& cluster, DPInstance* inst_1, DPInstance* inst_2)
{
  int32_t repeat_cnt = 0;
  for (auto* inst : cluster.get_inst_list()) {
    if (inst == inst_1 || inst == inst_2) {
      repeat_cnt += 1;
    }
  }
  return (repeat_cnt >= 2);
}

bool InstanceSwap::checkIfTwoClusterFusion2(DPCluster& cluster_1, DPCluster& cluster_2)
{
  std::string cluster1_name = cluster_1.get_name();
  std::string cluster2_name = cluster_2.get_name();
  std::string front1_name = "front1_name";
  std::string back1_name = "back1_name";
  std::string front2_name = "front2_name";
  std::string back2_name = "back2_name";

  if (cluster_1.get_front_cluster()) {
    front1_name = cluster_1.get_front_cluster()->get_name();
  }
  if (cluster_1.get_back_cluster()) {
    back1_name = cluster_1.get_back_cluster()->get_name();
  }
  if (cluster_2.get_front_cluster()) {
    front2_name = cluster_2.get_front_cluster()->get_name();
  }
  if (cluster_2.get_back_cluster()) {
    back2_name = cluster_2.get_back_cluster()->get_name();
  }

  bool flag_1 = (front1_name == front2_name || back1_name == back2_name);
  bool flag_2 = (front1_name == cluster2_name || back1_name == cluster2_name);
  bool flag_3 = (front2_name == cluster1_name || back2_name == cluster1_name);

  return (flag_1 || flag_2 || flag_3);
}

void InstanceSwap::eraseInstAndSplitCluster(DPCluster* cluster, DPInstance* inst)
{
  auto* target_interval = cluster->get_belong_interval();
  auto& inst_list = cluster->get_inst_list();

  std::vector<DPInstance*> target_list;
  for (size_t i = inst->get_internal_id() + 1; i < inst_list.size(); i++) {
    target_list.push_back(inst_list[i]);
  }

  if (!target_list.empty()) {
    std::string cluster_name = target_list[0]->get_name();
    if (_database->get_design()->find_cluster(cluster_name)) {
      cluster_name = cluster_name + "_plus";
    }

    DPCluster* new_cluster = new DPCluster(cluster_name);
    new_cluster->set_min_x(target_list[0]->get_coordi().get_x());
    new_cluster->set_belong_interval(target_interval);
    new_cluster->set_boundary_min_x(target_interval->get_min_x());
    new_cluster->set_boundary_max_x(target_interval->get_max_x());
    new_cluster->set_front_cluster(cluster);
    auto* back_cluster = cluster->get_back_cluster();
    new_cluster->set_back_cluster(back_cluster);
    if (back_cluster) {
      back_cluster->set_front_cluster(new_cluster);
    }

    for (size_t i = 0; i < target_list.size(); i++) {
      new_cluster->add_inst(target_list[i]);
      target_list[i]->set_belong_cluster(new_cluster);
      target_list[i]->set_internal_id(i);
    }

    _database->get_design()->add_cluster(new_cluster);
    cluster->set_back_cluster(new_cluster);
  }

  cluster->eraseInstanceRange(inst->get_internal_id(), inst_list.size() - 1);
  target_interval->updateRemainLength(inst->get_shape().get_width());
}

int64_t InstanceSwap::testCalTotalHPWL()
{
  int64_t total_hpwl = 0;
  for (auto* net : _database->get_design()->get_net_list()) {
    total_hpwl += net->calCurrentHPWL();
  }
  return total_hpwl;
}

}  // namespace ipl