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

#include "Abacus.hh"

namespace ipl {

Abacus::Abacus() : _database(nullptr), _config(nullptr), _row_height(-1), _site_width(-1)
{
}

Abacus::~Abacus()
{
  for (auto pair : _cluster_map) {
    delete pair.second;
  }
  _cluster_map.clear();
  _inst_belong_cluster.clear();
  _interval_cluster_root.clear();
  _interval_remain_length.clear();
}

void Abacus::initDataRequirement(LGConfig* lg_config, LGDatabase* lg_database)
{
  // clean abacus info first.
  _cluster_map.clear();
  _inst_belong_cluster.clear();
  _interval_cluster_root.clear();
  _interval_remain_length.clear();

  _database = lg_database;
  _config = lg_config;

  _inst_belong_cluster.resize(_database->get_lgInstance_list().size(), nullptr);

  int32_t interval_cnt = 0;
  for (auto& interval_vec : _database->get_lg_layout()->get_interval_2d_list()) {
    for (auto* interval : interval_vec) {
      interval_cnt++;
      _interval_remain_length.push_back(interval->get_max_x() - interval->get_min_x());
    }
  }
  _interval_cluster_root.resize(interval_cnt, nullptr);

  _row_height = _database->get_lg_layout()->get_row_height();
  _site_width = _database->get_lg_layout()->get_site_width();
}

bool Abacus::isInitialized()
{
  return !_cluster_map.empty();
}

void Abacus::specifyTargetInstList(std::vector<LGInstance*>& target_inst_list)
{
  _target_inst_list.clear();
  _target_inst_list = target_inst_list;
}

bool Abacus::runLegalization()
{
  // Sort all movable instances
  std::vector<LGInstance*> movable_inst_list;
  pickAndSortMovableInstList(movable_inst_list);

  int32_t inst_id = 0;
  for (auto* inst : movable_inst_list) {
    int32_t best_row = INT32_MAX;
    int32_t best_cost = INT32_MAX;
    for (int32_t row_idx = 0; row_idx < _database->get_lg_layout()->get_row_num(); row_idx++) {
      int32_t cost = placeRow(inst, row_idx, true);

      if (cost < best_cost) {
        best_cost = cost;
        best_row = row_idx;
      }
    }

    if (best_row == INT32_MAX) {
      LOG_ERROR << "Instance: " << inst->get_name() << "Cannot find a row for placement";
      return false;
    }

    placeRow(inst, best_row, false);

    inst_id++;
    if (inst_id % 100000 == 0) {
      LOG_INFO << "Place Instance : " << inst_id;
    }
  }

  return true;
}

bool Abacus::runIncrLegalization()
{
  int32_t row_range_num = 5;
  int32_t row_num = _database->get_lg_layout()->get_row_num();

  for (auto* inst : _target_inst_list) {
    int32_t row_idx = inst->get_coordi().get_y() / _row_height;
    int32_t max_row_idx = (row_idx + row_range_num > row_num) ? row_num : row_idx + row_range_num;
    int32_t min_row_idx = (row_idx - row_range_num < 0) ? 0 : row_idx - row_range_num;

    int32_t best_row = INT32_MAX;
    int32_t best_cost = INT32_MAX;
    for (int32_t row_idx = min_row_idx; row_idx < max_row_idx; row_idx++) {
      int32_t cost = placeRow(inst, row_idx, true);
      if (cost < best_cost) {
        best_cost = cost;
        best_row = row_idx;
      }
    }
    placeRow(inst, best_row, false);
  }

  return true;
}

void Abacus::pickAndSortMovableInstList(std::vector<LGInstance*>& movable_inst_list)
{
  for (auto* inst : _database->get_lgInstance_list()) {
    if (inst->get_state() == LGINSTANCE_STATE::kFixed) {
      continue;
    }
    movable_inst_list.push_back(inst);
  }

  std::sort(movable_inst_list.begin(), movable_inst_list.end(),
            [](LGInstance* l_inst, LGInstance* r_inst) { return (l_inst->get_coordi().get_x() < r_inst->get_coordi().get_x()); });
}

int32_t Abacus::placeRow(LGInstance* inst, int32_t row_idx, bool is_trial)
{
  Rectangle<int32_t> inst_shape = std::move(inst->get_shape());

  // Determine clusters and their optimal positions x_c(c):
  std::vector<LGInterval*> interval_list = _database->get_lg_layout()->get_interval_2d_list()[row_idx];

  // Select the nearest interval for the instance
  int32_t row_interval_idx = searchNearestIntervalIndex(interval_list, inst_shape);
  if (row_interval_idx == INT32_MAX) {
    // LOG_WARNING << "Instance is not overlap with interval!";
    return INT32_MAX;
  }

  if (_interval_remain_length[interval_list[row_interval_idx]->get_index()] < inst_shape.get_width()) {
    // Select the most recent non-full interval
    int32_t origin_idx = row_interval_idx;
    row_interval_idx = searchRemainSpaceSegIndex(interval_list, inst_shape, origin_idx);
    if (row_interval_idx == origin_idx) {
      // LOG_INFO << "Row : " << row_idx << " has no room to place.";
      return INT32_MAX;
    }
  }

  // Arrange inst into interval
  auto* target_interval = interval_list[row_interval_idx];
  AbacusCluster target_cluster = std::move(arrangeInstIntoIntervalCluster(inst, target_interval));

  // Calculate cost
  int32_t movement_cost = 0;
  int32_t coordi_x = target_cluster.get_min_x();
  int32_t inst_movement_x = INT32_MAX;
  for (auto* target_inst : target_cluster.get_inst_list()) {
    int32_t origin_x = target_inst->get_coordi().get_x();
    if (target_inst == inst) {
      inst_movement_x = std::abs(coordi_x - origin_x);
    } else {
      movement_cost += std::abs(coordi_x - origin_x);
    }
    coordi_x += target_inst->get_shape().get_width();
  }
  // Add Inst Coordi Movement Cost
  int32_t inst_movement_y = std::abs(row_idx * _row_height - inst_shape.get_ll_y());
  int32_t inst_displacement = inst_movement_x + inst_movement_y;
  movement_cost += inst_displacement;

  // Penalize violations of maximum movement constraints
  if (inst_displacement > _config->get_max_displacement()) {
    movement_cost += _database->get_lg_layout()->get_max_x();
  }

  // Replace cluster
  if (!is_trial) {
    replaceClusterInfo(target_cluster);
    this->updateRemainLength(target_interval, -(inst->get_shape().get_width()));
  }

  return movement_cost;
}

int32_t Abacus::searchNearestIntervalIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape)
{
  if (segment_list.size() == 1) {
    return 0;
  }

  int32_t prev_distance = INT32_MAX;
  int32_t segment_idx = INT32_MAX;
  for (size_t i = 0; i < segment_list.size(); i++) {
    int32_t cur_distance
        = calDistanceWithBox(inst_shape.get_ll_x(), inst_shape.get_ur_x(), segment_list[i]->get_min_x(), segment_list[i]->get_max_x());
    if (cur_distance > prev_distance) {
      segment_idx = i - 1;
      break;
    }
    if (cur_distance == 0) {
      segment_idx = i;
      break;
    }

    prev_distance = cur_distance;
  }

  return segment_idx;
}

int32_t Abacus::calDistanceWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x)
{
  if (max_x >= box_min_x && min_x <= box_max_x) {
    return 0;
  } else if (min_x > box_max_x) {
    return (min_x - box_max_x);
  } else if (max_x < box_min_x) {
    return (box_min_x - max_x);
  } else {
    return INT32_MAX;
  }
}

int32_t Abacus::searchRemainSpaceSegIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape, int32_t origin_index)
{
  int32_t segment_idx = origin_index;
  // int32_t max_range = segment_list.size() - 1;
  int32_t max_range = 2;
  int32_t range = 1;
  while (range <= max_range) {
    int32_t r_idx = segment_idx;
    if (r_idx + range < static_cast<int32_t>(segment_list.size())) {
      r_idx += range;
      int32_t interval_idx = segment_list[r_idx]->get_index();
      if (_interval_remain_length[interval_idx] >= inst_shape.get_width()) {
        segment_idx = r_idx;
        break;
      }
    }
    int32_t l_idx = segment_idx;
    if (l_idx - range >= 0) {
      l_idx -= range;
      int32_t interval_idx = segment_list[l_idx]->get_index();
      if (_interval_remain_length[interval_idx] >= inst_shape.get_width()) {
        segment_idx = l_idx;
        break;
      }
    }
    range++;
  }
  return segment_idx;
}

AbacusCluster Abacus::arrangeInstIntoIntervalCluster(LGInstance* inst, LGInterval* interval)
{
  auto inst_shape = std::move(inst->get_shape());
  AbacusCluster record_cluster;
  auto* cur_cluster = _interval_cluster_root[interval->get_index()];
  auto* last_cluster = cur_cluster;
  bool is_collapse = false;

  if (cur_cluster && (inst_shape.get_ur_x() < cur_cluster->get_min_x())) {
    // should insert in the cluster
    record_cluster = *cur_cluster;
    record_cluster.insertInstance(inst);
    legalizeCluster(record_cluster);
    is_collapse = true;
  } else {
    while (cur_cluster) {
      if (checkOverlapWithBox(inst_shape.get_ll_x(), inst_shape.get_ur_x(), cur_cluster->get_min_x(), cur_cluster->get_max_x())) {
        record_cluster = *cur_cluster;
        record_cluster.insertInstance(inst);
        legalizeCluster(record_cluster);
        is_collapse = true;
        break;
      }
      auto* back_cluster = cur_cluster->get_back_cluster();
      if (back_cluster) {
        // tmp fix bug.
        if (inst_shape.get_ll_x() >= cur_cluster->get_max_x() && inst_shape.get_ur_x() <= back_cluster->get_min_x()) {
          record_cluster = *cur_cluster;
          record_cluster.insertInstance(inst);
          legalizeCluster(record_cluster);
          is_collapse = true;
          break;
        }

        last_cluster = back_cluster;
      }
      cur_cluster = back_cluster;
    }
  }

  if (!is_collapse) {
    // Create new cluster
    record_cluster = AbacusCluster(inst->get_name());
    record_cluster.add_inst(inst);
    record_cluster.updateAbacusInfo(inst);
    record_cluster.set_belong_interval(interval);
    record_cluster.set_front_cluster(last_cluster);
    legalizeCluster(record_cluster);
  }

  return record_cluster;
}

void Abacus::legalizeCluster(AbacusCluster& cluster)
{
  arrangeClusterMinXCoordi(cluster);
  int32_t cur_min_x, front_max_x, back_min_x;
  cur_min_x = cluster.get_min_x();
  front_max_x = obtainFrontMaxX(cluster);
  while (cur_min_x < front_max_x) {
    AbacusCluster front_cluster = *(cluster.get_front_cluster());
    front_cluster.appendCluster(cluster);
    front_cluster.set_back_cluster(cluster.get_back_cluster());
    arrangeClusterMinXCoordi(front_cluster);
    cur_min_x = front_cluster.get_min_x();
    front_max_x = obtainFrontMaxX(front_cluster);
    cluster = front_cluster;
  }

  cur_min_x = cluster.get_min_x();
  back_min_x = obtainBackMinX(cluster);
  while (cur_min_x + cluster.get_total_width() > back_min_x) {
    auto* back_cluster = cluster.get_back_cluster();
    cluster.appendCluster(*back_cluster);
    cluster.set_back_cluster(back_cluster->get_back_cluster());
    arrangeClusterMinXCoordi(cluster);
    cur_min_x = cluster.get_min_x();
    back_min_x = obtainBackMinX(cluster);
  }
}

void Abacus::arrangeClusterMinXCoordi(AbacusCluster& cluster)
{
  int32_t cluster_x = (cluster.get_weight_q() / cluster.get_weight_e());
  cluster_x = (cluster_x / _site_width) * _site_width;

  int32_t boundary_min_x = cluster.get_belong_interval()->get_min_x();
  int32_t boundary_max_x = cluster.get_belong_interval()->get_max_x();
  cluster_x < boundary_min_x ? cluster_x = boundary_min_x : cluster_x;
  cluster_x + cluster.get_total_width() > boundary_max_x ? cluster_x = boundary_max_x - cluster.get_total_width() : cluster_x;

  if (cluster_x < boundary_min_x) {
    LOG_WARNING << "Cluster width is out of interval capcity";
  }

  cluster.set_min_x(cluster_x);
}

int32_t Abacus::obtainFrontMaxX(AbacusCluster& cluster)
{
  int32_t front_max_x = cluster.get_belong_interval()->get_min_x();
  if (cluster.get_front_cluster()) {
    front_max_x = cluster.get_front_cluster()->get_max_x();
  }
  return front_max_x;
}

int32_t Abacus::obtainBackMinX(AbacusCluster& cluster)
{
  int32_t back_min_x = cluster.get_belong_interval()->get_max_x();
  if (cluster.get_back_cluster()) {
    back_min_x = cluster.get_back_cluster()->get_min_x();
  }
  return back_min_x;
}

bool Abacus::checkOverlapWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x)
{
  if (max_x > box_min_x && min_x < box_max_x) {
    return true;
  } else {
    return false;
  }
}

void Abacus::replaceClusterInfo(AbacusCluster& modify_cluster)
{
  auto* origin_interval = modify_cluster.get_belong_interval();
  int32_t coordi_y = origin_interval->get_belong_row()->get_coordinate().get_y();

  auto* cluster_ptr = this->findCluster(modify_cluster.get_name());
  if (!cluster_ptr) {
    AbacusCluster* new_cluster = new AbacusCluster(std::move(modify_cluster));
    auto inst_list = new_cluster->get_inst_list();
    if (inst_list.size() > 1 || inst_list.size() == 0) {
      LOG_WARNING << "Cluster Inst is not correctly set";
    }

    // cluster setting.
    _inst_belong_cluster[inst_list[0]->get_index()] = new_cluster;
    inst_list[0]->updateCoordi(new_cluster->get_min_x(), coordi_y);

    this->insertCluster(new_cluster->get_name(), new_cluster);

    if (!_interval_cluster_root[origin_interval->get_index()]) {
      _interval_cluster_root[origin_interval->get_index()] = new_cluster;
    }

    if (new_cluster->get_front_cluster()) {
      new_cluster->get_front_cluster()->set_back_cluster(new_cluster);
    }
    return;
  }

  auto& origin_cluster = *(cluster_ptr);
  auto* origin_root = _interval_cluster_root[origin_interval->get_index()];

  // may be collapsing with front or back cluster
  AbacusCluster* front_origin = origin_cluster.get_front_cluster();
  AbacusCluster* front_modify = modify_cluster.get_front_cluster();
  while (front_origin != front_modify) {
    if (front_origin == origin_root) {
      _interval_cluster_root[origin_interval->get_index()] = &origin_cluster;
    }

    std::string delete_cluster = front_origin->get_name();
    front_origin = front_origin->get_front_cluster();
    if (front_origin) {
      front_origin->set_back_cluster(&origin_cluster);
    }
    this->deleteCluster(delete_cluster);
  }

  AbacusCluster* back_origin = origin_cluster.get_back_cluster();
  AbacusCluster* back_modify = modify_cluster.get_back_cluster();
  while (back_origin != back_modify) {
    std::string delete_cluster = back_origin->get_name();

    back_origin = back_origin->get_back_cluster();
    if (back_origin) {
      back_origin->set_front_cluster(&origin_cluster);
    }
    this->deleteCluster(delete_cluster);
  }

  // update all inst info
  origin_cluster = std::move(modify_cluster);
  int32_t coordi_x = origin_cluster.get_min_x();
  for (auto* inst : origin_cluster.get_inst_list()) {
    _inst_belong_cluster[inst->get_index()] = &origin_cluster;
    inst->updateCoordi(coordi_x, coordi_y);
    coordi_x += inst->get_shape().get_width();
  }
}

AbacusCluster* Abacus::findCluster(std::string cluster_name)
{
  AbacusCluster* cluster = nullptr;
  auto it = _cluster_map.find(cluster_name);
  if (it != _cluster_map.end()) {
    cluster = it->second;
  }

  return cluster;
}

void Abacus::insertCluster(std::string name, AbacusCluster* cluster)
{
  auto it = _cluster_map.find(name);
  if (it != _cluster_map.end()) {
    LOG_WARNING << "Cluster : " << name << " was added before";
  }
  _cluster_map.emplace(name, cluster);
}

void Abacus::deleteCluster(std::string name)
{
  auto it = _cluster_map.find(name);
  if (it != _cluster_map.end()) {
    _cluster_map.erase(it);
  } else {
    LOG_WARNING << "Cluster: " << name << " has not been insert";
  }
}

void Abacus::updateRemainLength(LGInterval* interval, int32_t delta)
{
  int32_t cur_value = _interval_remain_length[interval->get_index()];
  _interval_remain_length[interval->get_index()] = cur_value + delta;
}

}  // namespace ipl