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
#include "BinOpt.hh"

#include "utility/Utility.hh"

namespace ipl {

BinOpt::BinOpt(DPConfig* config, DPDatabase* database, DPOperator* dp_operator)
{
  _config = config;
  _database = database;
  _row_height = database->get_layout()->get_row_height();
  _site_width = database->get_layout()->get_site_width();
  _operator = dp_operator;
}

BinOpt::~BinOpt()
{
}

void BinOpt::runBinOpt()
{
  bool is_clusted = _operator->checkIfClustered();
  if (!is_clusted) {
    _operator->updateInstClustering();
  }

  auto* grid_manager = _operator->get_grid_manager();
  int64_t grid_size_x = grid_manager->get_grid_size_x();
  int64_t grid_size_y = grid_manager->get_grid_size_y();
  int64_t grid_area = grid_size_x * grid_size_y;

  // update grid_manager
  _operator->updateGridManager();

  // left to right
  auto& grid_2d_list = grid_manager->get_grid_2d_list();
  for (auto& grid_row : grid_2d_list) {
    for (size_t i = 0, j = i + 1; i < grid_row.size() && j < grid_row.size(); i++, j++) {
      auto* supply_grid = &grid_row[i];
      auto* demand_grid = &grid_row[j];
      slidingInstBetweenGrids(supply_grid, demand_grid, grid_area);
    }
  }

  // // left to right
  // for (auto* grid_row : grid_manager->get_row_list()) {
  //   auto& grid_list = grid_row->get_grid_list();
  //   for (size_t i = 0, j = i + 1; i < grid_list.size() && j < grid_list.size(); i++, j++) {
  //     auto* supply_grid = grid_list[i];
  //     auto* demand_grid = grid_list[j];
  //     slidingInstBetweenGrids(supply_grid, demand_grid, grid_area);
  //   }
  // }

  // update grid_manager
  _operator->updateGridManager();

  // right to left
  for (auto& grid_row : grid_2d_list) {
    for (int32_t i = grid_row.size() - 1, j = i - 1; i >= 0 && j >= 0; i--, j--) {
      auto* supply_grid = &grid_row[i];
      auto* demand_grid = &grid_row[j];
      slidingInstBetweenGrids(supply_grid, demand_grid, grid_area);
    }
  }

  // // right to left
  // for (auto* grid_row : grid_manager->get_row_list()) {
  //   auto& grid_list = grid_row->get_grid_list();
  //   for (int32_t i = grid_list.size() - 1, j = i - 1; i >= 0 && j >= 0; i--, j--) {
  //     auto* supply_grid = grid_list[i];
  //     auto* demand_grid = grid_list[j];
  //     slidingInstBetweenGrids(supply_grid, demand_grid, grid_area);
  //   }
  // }
}

void BinOpt::slidingInstBetweenGrids(Grid* supply_grid, Grid* demand_grid, int64_t grid_area)
{
  int64_t target_area = static_cast<float>(grid_area) * supply_grid->available_ratio;

  if (supply_grid->occupied_area < target_area) {
    return;
  }
  if (supply_grid->occupied_area < demand_grid->occupied_area) {
    return;
  }

  // int64_t target_area = static_cast<float>(grid_area) * supply_grid->get_available_ratio();

  // if (supply_grid->get_occupied_area() < target_area) {
  //   return;
  // }
  // if (supply_grid->get_occupied_area() < demand_grid->get_occupied_area()) {
  //   return;
  // }

  int64_t flow_value = calSlidingFlowValue(supply_grid, demand_grid, target_area);

  if (flow_value <= 0) {
    return;
  }
  Utility utility;

  auto supply_grid_shape = supply_grid->shape;
  auto demand_grid_shape = demand_grid->shape;

  // auto supply_grid_shape = std::move(supply_grid->get_shape());
  // auto demand_grid_shape = std::move(demand_grid->get_shape());

  std::pair<int32_t, int32_t> row_range
      = utility.obtainMinMaxIdx(0, _row_height, supply_grid_shape.get_ll_y(), supply_grid_shape.get_ur_y());
  int64_t row_avg_flow = flow_value / (row_range.second - row_range.first);

  bool lr_flag = true;
  int32_t grid_cut_x = INT32_MIN;
  if (supply_grid_shape.get_ur_x() == demand_grid_shape.get_ll_x()) {
    lr_flag = true;
    grid_cut_x = supply_grid_shape.get_ur_x();
  } else if (supply_grid_shape.get_ll_x() == demand_grid_shape.get_ur_x()) {
    lr_flag = false;
    grid_cut_x = supply_grid_shape.get_ll_x();
  }

  auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();
  for (int i = row_range.first; i < row_range.second; i++) {
    auto& interval_list = interval_2d_list[i];
    int32_t right_index = INT32_MIN;
    for (size_t j = 0; j < interval_list.size(); j++) {
      int32_t interval_max_x = interval_list[j]->get_max_x();
      if (interval_max_x > grid_cut_x) {
        right_index = j;
        break;
      }
    }

    if (right_index > 0) {
      if (lr_flag) {
        auto* target_interval = interval_list[right_index];
        auto* source_interval = interval_list[right_index - 1];

        auto* last_cluster = obtainIntervalLastCluster(source_interval);

        int32_t shift_area = 0;
        while (shift_area < row_avg_flow && last_cluster) {
          auto& inst_list = last_cluster->get_inst_list();
          std::set<int32_t, std::greater<int32_t>> delete_indexes;
          auto rit = inst_list.rbegin();

          bool continue_flag = true;
          while (rit != inst_list.rend()) {
            int32_t inst_index = (*rit)->get_internal_id();
            int64_t inst_width = (*rit)->get_shape().get_width();
            if (inst_width > target_interval->get_remain_length()) {
              continue_flag = false;
              break;
            }

            if (moveInstToInterval(*rit, target_interval)) {
              delete_indexes.emplace(inst_index);
              // last_cluster->eraseInstance(inst_index);
              source_interval->updateRemainLength(inst_width);
              int64_t inst_height = (*rit)->get_shape().get_height();
              shift_area += (inst_width * inst_height);
            }

            if (shift_area >= row_avg_flow) {
              continue_flag = false;
              break;
            }
            ++rit;
          }

          // tmp fix bug.
          for (int32_t index : delete_indexes) {
            last_cluster->eraseInstance(index);
          }

          if (!continue_flag) {
            break;
          }

          last_cluster = last_cluster->get_front_cluster();
        }
      } else {
        auto* target_interval = interval_list[right_index - 1];
        auto* source_interval = interval_list[right_index];
        auto* front_cluster = obtainIntervalFirstCluster(source_interval);

        int32_t shift_area = 0;
        while (shift_area < row_avg_flow && front_cluster) {
          auto& inst_list = front_cluster->get_inst_list();
          std::set<int32_t, std::greater<int32_t>> delete_indexs;
          bool continue_flag = true;
          auto it = inst_list.begin();
          while (it != inst_list.end()) {
            int32_t inst_index = (*it)->get_internal_id();
            int64_t inst_width = (*it)->get_shape().get_width();
            if (inst_width > target_interval->get_remain_length()) {
              continue_flag = false;
              break;
            }

            if (moveInstToInterval(*it, target_interval)) {
              delete_indexs.emplace(inst_index);
              // front_cluster->eraseInstance(inst_index);
              source_interval->updateRemainLength(inst_width);
              int64_t inst_height = (*it)->get_shape().get_height();
              shift_area += (inst_width * inst_height);
            }

            if (shift_area >= row_avg_flow) {
              continue_flag = false;
              break;
            }

            ++it;
          }

          // tmp fix bug.
          for (int32_t index : delete_indexs) {
            front_cluster->eraseInstance(index);
          }

          if (!continue_flag) {
            break;
          }

          front_cluster = front_cluster->get_front_cluster();
        }
      }
    }
  }
}

int64_t BinOpt::calSlidingFlowValue(Grid* supply_grid, Grid* demand_grid, int64_t target_area)
{
  int64_t flow_value = 0;

  int64_t supply_area = supply_grid->occupied_area - target_area;
  int64_t demand_area = target_area - demand_grid->occupied_area;
  // int64_t supply_area = supply_grid->get_occupied_area() - target_area;
  // int64_t demand_area = target_area - demand_grid->get_occupied_area();

  if (demand_area > 0) {
    if (supply_area - demand_area <= 0) {
      flow_value = supply_area;
    } else {
      flow_value = (supply_area - demand_area) / 2;
    }
  } else {
    flow_value = (supply_area + demand_area) / 2;
  }

  return flow_value;
}

DPCluster* BinOpt::obtainIntervalFirstCluster(DPInterval* interval)
{
  return interval->get_cluster_root();
}

DPCluster* BinOpt::obtainIntervalLastCluster(DPInterval* interval)
{
  DPCluster* cur_cluster = interval->get_cluster_root();
  DPCluster* last_cluster = cur_cluster;
  while (cur_cluster) {
    if (cur_cluster->get_back_cluster()) {
      last_cluster = cur_cluster->get_back_cluster();
    }
    cur_cluster = cur_cluster->get_back_cluster();
  }
  return last_cluster;
}

bool BinOpt::moveInstToInterval(DPInstance* inst, DPInterval* interval)
{
  auto inst_shape = std::move(inst->get_shape());
  if (inst_shape.get_width() > interval->get_remain_length()) {
    return false;
  }

  if (inst_shape.get_ll_x() > interval->get_max_x()) {
    // interval in left side
    auto* last_cluster = obtainIntervalLastCluster(interval);
    if (last_cluster) {
      inst->set_internal_id(last_cluster->get_inst_list().size());
      last_cluster->add_inst(inst);
      inst->set_belong_cluster(last_cluster);
    } else {
      last_cluster = createInstClusterForInterval(inst, interval);
      last_cluster->set_min_x(interval->get_max_x() - inst_shape.get_width());
    }
    instantLegalizeCluster(last_cluster);
  } else {
    // interval in right side
    auto* front_cluster = obtainIntervalFirstCluster(interval);
    if (front_cluster) {
      front_cluster->insertInstance(inst, 0);
      inst->set_belong_cluster(front_cluster);
    } else {
      front_cluster = createInstClusterForInterval(inst, interval);
      front_cluster->set_min_x(interval->get_min_x());
    }
    instantLegalizeCluster(front_cluster);
  }
  interval->updateRemainLength(-inst_shape.get_width());

  return true;
}

void BinOpt::instantLegalizeCluster(DPCluster* cluster)
{
  arrangeClusterMinXCoordi(cluster);
  int32_t cur_min_x, front_max_x, back_min_x;
  cur_min_x = cluster->get_min_x();
  front_max_x = obtainFrontMaxX(cluster);
  while (cur_min_x < front_max_x) {
    auto* front_cluster = cluster->get_front_cluster();
    collapseCluster(front_cluster, cluster);
    arrangeClusterMinXCoordi(front_cluster);
    cur_min_x = front_cluster->get_min_x();
    front_max_x = obtainFrontMaxX(front_cluster);
    cluster = front_cluster;
  }

  cur_min_x = cluster->get_min_x();
  back_min_x = obtainBackMinX(cluster);
  while (cur_min_x + cluster->get_total_width() > back_min_x) {
    auto* back_cluster = cluster->get_back_cluster();
    if (!back_cluster) {
      //
    }
    collapseCluster(cluster, back_cluster);
    arrangeClusterMinXCoordi(cluster);
    cur_min_x = cluster->get_min_x();
    back_min_x = obtainBackMinX(cluster);
  }
}

void BinOpt::arrangeClusterMinXCoordi(DPCluster* cluster)
{
  double weight_e = 0.0;
  double weight_q = 0.0;
  double total_width = 0;

  int32_t coordi_x = cluster->get_min_x();
  for (auto* inst : cluster->get_inst_list()) {
    weight_e += inst->get_weight();
    weight_q += inst->get_weight() * (coordi_x - total_width);
    int32_t inst_width = inst->get_shape().get_width();
    total_width += inst_width;
    coordi_x += inst_width;
  }

  int32_t cluster_x = weight_q / weight_e;

  cluster_x = (cluster_x / _site_width) * _site_width;
  cluster_x < cluster->get_boundary_min_x() ? cluster_x = cluster->get_boundary_min_x() : cluster_x;
  cluster_x + total_width > cluster->get_boundary_max_x() ? cluster_x = cluster->get_boundary_max_x() - total_width : cluster_x;

  cluster->set_min_x(cluster_x);

  // instant set inst coordi
  int32_t y_coordi = cluster->get_belong_interval()->get_belong_row()->get_coordinate().get_y();
  int32_t x_coordi = cluster->get_min_x();
  for (auto* inst : cluster->get_inst_list()) {
    inst->updateCoordi(x_coordi, y_coordi);
    x_coordi += inst->get_shape().get_width();
  }
}

int32_t BinOpt::obtainFrontMaxX(DPCluster* cluster)
{
  int32_t front_max_x = cluster->get_boundary_min_x();
  if (cluster->get_front_cluster()) {
    front_max_x = cluster->get_front_cluster()->get_max_x();
  }
  return front_max_x;
}

int32_t BinOpt::obtainBackMinX(DPCluster* cluster)
{
  int32_t back_min_x = cluster->get_boundary_max_x();
  if (cluster->get_back_cluster()) {
    back_min_x = cluster->get_back_cluster()->get_min_x();
  }
  return back_min_x;
}

void BinOpt::collapseCluster(DPCluster* dest_cluster, DPCluster* src_cluster)
{
  const auto& inst_list = src_cluster->get_inst_list();
  for (auto* inst : inst_list) {
    dest_cluster->add_inst(inst);
  }
  dest_cluster->set_back_cluster(src_cluster->get_back_cluster());
  if (src_cluster->get_back_cluster()) {
    src_cluster->get_back_cluster()->set_front_cluster(dest_cluster);
  }

  // update inst belong cluster and set the internal id
  int32_t inst_internal_id = 0;
  for (auto* inst : dest_cluster->get_inst_list()) {
    inst->set_belong_cluster(dest_cluster);
    inst->set_internal_id(inst_internal_id);
    inst_internal_id++;
  }

  // delete src_cluster
  _database->get_design()->deleteCluster(src_cluster->get_name());
}

DPCluster* BinOpt::createInstClusterForInterval(DPInstance* inst, DPInterval* interval)
{
  DPCluster* cluster = nullptr;
  if (!interval) {
    LOG_WARNING << "Cannot find the interval for current inst position";
    return cluster;
  }

  if (interval->get_cluster_root()) {
    LOG_WARNING << "Current interval already has root cluster";
    return cluster;
  }

  int32_t interval_min_x = interval->get_min_x();
  int32_t interval_max_x = interval->get_max_x();

  cluster = new DPCluster(inst->get_name() + "Shift_" + interval->get_name());
  cluster->set_min_x(interval_min_x);
  cluster->add_inst(inst);
  cluster->set_belong_interval(interval);
  cluster->set_boundary_min_x(interval_min_x);
  cluster->set_boundary_max_x(interval_max_x);
  inst->set_belong_cluster(cluster);
  inst->set_internal_id(0);

  interval->set_cluster_root(cluster);
  _database->get_design()->add_cluster(cluster);

  return cluster;
}

}  // namespace ipl