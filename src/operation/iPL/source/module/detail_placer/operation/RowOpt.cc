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
#include "RowOpt.hh"

#include "module/logger/Log.hh"

namespace ipl {

RowOpt::RowOpt(DPConfig* config, DPDatabase* database, DPOperator* dp_operator)
    : _config(config), _database(database), _operator(dp_operator)
{
  _site_width = database->get_layout()->get_site_width();
  updateIntervalInfo();
}

RowOpt::~RowOpt()
{
}

void RowOpt::updateIntervalInfo()
{
  // clear
  resetAllInterval();
  clearCluster();

  // obatain movable inst list
  std::vector<DPInstance*> movable_inst_list;
  pickAndSortMovableInstList(movable_inst_list);

  // create cluster for every movable inst
  convertInstListToClusters(movable_inst_list);

  // update cluster bound list
  //   updateClusterBoundList();
}

void RowOpt::pickAndSortMovableInstList(std::vector<DPInstance*>& movable_inst_list)
{
  const std::vector<DPInstance*>& inst_list = _database->get_design()->get_inst_list();
  movable_inst_list.reserve(inst_list.size());
  // obatain movable inst list
  for (auto* dp_inst : inst_list) {
    if (dp_inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }
    movable_inst_list.push_back(dp_inst);
  }
  std::sort(movable_inst_list.begin(), movable_inst_list.end(),
            [](DPInstance* l_inst, DPInstance* r_inst) { return (l_inst->get_coordi().get_x() < r_inst->get_coordi().get_x()); });
}

void RowOpt::convertInstListToClusters(std::vector<DPInstance*>& movable_inst_list)
{
  const std::vector<std::vector<DPInterval*>>& interval_list = _database->get_layout()->get_interval_2d_list();
  int32_t row_height = _database->get_layout()->get_row_height();

  // tmp for clusters connection
  std::map<DPInterval*, DPCluster*> interval_last_cluster;

  for (auto* inst : movable_inst_list) {
    int32_t inst_lx = inst->get_shape().get_ll_x();
    int32_t inst_ux = inst->get_shape().get_ur_x();
    int32_t row_index = inst->get_coordi().get_y() / row_height;
    for (auto* interval : interval_list[row_index]) {
      if (interval->checkInLine(inst_lx, inst_ux)) {
        DPCluster* cluster = createCluster(inst, interval);
        DPCluster* front_cluster = nullptr;

        auto it = interval_last_cluster.find(interval);
        if (it != interval_last_cluster.end()) {
          front_cluster = it->second;
          it->second = cluster;
        } else {
          interval_last_cluster.emplace(interval, cluster);
          _interval_to_root.emplace(interval, cluster);
        }

        if (front_cluster) {
          cluster->set_front_cluster(front_cluster);
          front_cluster->set_back_cluster(cluster);
        }
        storeCluster(cluster);
        break;
      }
    }
  }
}

void RowOpt::updateClusterBoundList()
{
  for (auto pair : _interval_to_root) {
    DPCluster* cluster_root = pair.second;
    DPCluster* current_cluster = cluster_root;

    while (current_cluster) {
      std::vector<int32_t> bound_list;
      generateClusterBounds(current_cluster, bound_list);
      current_cluster->add_bound_list(bound_list);
      current_cluster = current_cluster->get_back_cluster();
    }
  }
}

void RowOpt::generateClusterBounds(DPCluster* cluster, std::vector<int32_t>& bound_list)
{
  int32_t core_max_x = _database->get_layout()->get_max_x();

  auto* current_interval = cluster->get_belong_interval();
  for (auto* inst : cluster->get_inst_list()) {
    int32_t inst_x = inst->get_coordi().get_x();

    for (auto* inst_pin : inst->get_pin_list()) {
      std::pair<int32_t, int32_t> pin_offset = std::move(inst->calInstPinModifyOffest(inst_pin));

      auto* pin_net = inst_pin->get_net();
      DPPin* l_pin = nullptr;
      DPPin* r_pin = nullptr;
      int32_t max_x = INT32_MIN;
      int32_t min_x = INT32_MAX;
      for (auto* outpin : pin_net->get_pins()) {
        if (outpin == inst_pin) {
          continue;
        }
        if (outpin->get_x_coordi() < min_x) {
          l_pin = outpin;
          min_x = outpin->get_x_coordi();
        }
        if (outpin->get_x_coordi() > max_x) {
          r_pin = outpin;
          max_x = outpin->get_x_coordi();
        }
      }
      // l_pin and r_pin are both bounds
      std::set<DPPin*> bound_pins{l_pin, r_pin};
      for (auto* bound_pin : bound_pins) {
        if (!bound_pin) {
          continue;
        }

        int32_t bound_pin_x = bound_pin->get_x_coordi();
        auto* bound_pin_inst = bound_pin->get_instance();

        if (bound_pin_inst) {
          auto* bound_cluster = bound_pin_inst->get_belong_cluster();
          if (bound_cluster) {
            if (bound_cluster->get_belong_interval() == current_interval) {
              if (bound_pin_x < inst_x) {
                bound_list.push_back(current_interval->get_min_x());
              } else {
                bound_list.push_back(current_interval->get_max_x());
              }
              continue;
            }
          }
        }

        int32_t bound_lx = bound_pin_x - pin_offset.first - (inst->get_master()->get_width() / 2);
        if (bound_lx < 0) {
          bound_lx = 0;
        }
        if (bound_lx + cluster->get_total_width() > core_max_x) {
          bound_lx = core_max_x - cluster->get_total_width();
        }
        bound_list.push_back(bound_lx);
      }
    }
  }
}

DPCluster* RowOpt::createCluster(DPInstance* inst, DPInterval* interval)
{
  DPCluster* cluster = new DPCluster(inst->get_name());
  cluster->set_belong_interval(interval);
  cluster->set_boundary_min_x(interval->get_min_x());
  cluster->set_boundary_max_x(interval->get_max_x());
  cluster->add_inst(inst);
  inst->set_internal_id(0);
  inst->set_belong_cluster(cluster);

  return cluster;
}

void RowOpt::clearCluster()
{
  auto* design = _database->get_design();
  design->clearClusterInfo();
  _interval_to_root.clear();
}

void RowOpt::resetAllInterval()
{
  _database->get_layout()->resetAllInterval();
}

void RowOpt::storeCluster(DPCluster* cluster)
{
  auto* design = _database->get_design();
  design->add_cluster(cluster);
}

void RowOpt::deleteCluster(DPCluster* cluster)
{
  auto* design = _database->get_design();
  design->deleteCluster(cluster->get_name());
}

void RowOpt::correctOptimalLineInInterval(std::pair<int32_t, int32_t>& optimal_line, DPInterval* interval, int32_t width)
{
  if (optimal_line.first < interval->get_min_x()) {
    optimal_line.first = interval->get_min_x();
  }
  if (optimal_line.first + width > interval->get_max_x()) {
    optimal_line.first = interval->get_max_x() - width;
  }

  if (optimal_line.second < interval->get_min_x()) {
    optimal_line.second = interval->get_min_x();
  }
  if (optimal_line.second + width > interval->get_max_x()) {
    optimal_line.second = interval->get_max_x() - width;
  }
}

void RowOpt::runRowOpt()
{
  int32_t interval_count = 0;
  for (auto pair : _interval_to_root) {
    DPInterval* interval = pair.first;
    DPCluster* cluster_root = pair.second;

    if (!cluster_root) {
      continue;
    }

    DPCluster* current_cluster = cluster_root;

    while (current_cluster) {
      std::vector<int32_t> bound_list;
      generateClusterBounds(current_cluster, bound_list);
      current_cluster->add_bound_list(bound_list);

      std::pair<int32_t, int32_t> optimal_line = std::move(current_cluster->obtainOptimalMinCoordiLine());

      correctOptimalLineInInterval(optimal_line, interval, current_cluster->get_total_width());
      auto* front_cluster = current_cluster->get_front_cluster();
      auto* back_cluster = current_cluster->get_back_cluster();
      if (front_cluster) {
        int32_t front_max_x = front_cluster->get_max_x();
        if (optimal_line.second >= front_max_x) {
          int32_t optimal_x = optimal_line.first > front_max_x ? optimal_line.first : front_max_x;
          current_cluster->set_min_x(obtainOptimalLegalCoordiX(optimal_x, optimal_line));
        } else {  // overlap
          current_cluster->set_min_x(optimal_line.second);
          collapseClusters(front_cluster, current_cluster);
        }
      } else {
        current_cluster->set_min_x(obtainOptimalLegalCoordiX(optimal_line.first, optimal_line));
      }
      current_cluster = back_cluster;
    }

    DPCluster* cluster_record = cluster_root;
    int32_t coordi_y = interval->get_belong_row()->get_coordinate().get_y();
    while (cluster_record) {
      int32_t coordi_x = cluster_record->get_min_x();
      int32_t internal_id = 0;
      for (auto* inst : cluster_record->get_inst_list()) {
        inst->set_belong_cluster(cluster_record);
        inst->set_internal_id(internal_id++);
        inst->updateCoordi(coordi_x, coordi_y);

        int32_t inst_width = inst->get_shape().get_width();
        coordi_x += inst_width;
        interval->updateRemainLength(0 - inst_width);
      }
      cluster_record = cluster_record->get_back_cluster();
    }
    interval->set_cluster_root(cluster_root);

    ++interval_count;
  }
}

void RowOpt::collapseClusters(DPCluster* dest_cluster, DPCluster* src_cluster)
{
  if (!dest_cluster) {
    return;
  }

  int32_t dest_max_x = dest_cluster->get_max_x();
  int32_t src_min_x = src_cluster->get_min_x();
  if (dest_max_x <= src_min_x) {
    return;
  } else {
    dest_cluster->appendCluster(src_cluster);
    deleteCluster(src_cluster);
    std::pair<int32_t, int32_t> optimal_line = dest_cluster->obtainOptimalMinCoordiLine();
    correctOptimalLineInInterval(optimal_line, dest_cluster->get_belong_interval(), dest_cluster->get_total_width());

    auto* front_cluster = dest_cluster->get_front_cluster();
    if (front_cluster) {
      int32_t front_max_x = front_cluster->get_max_x();
      if (optimal_line.second >= front_max_x) {
        int32_t optimal_x = optimal_line.first > front_max_x ? optimal_line.first : front_max_x;
        dest_cluster->set_min_x(obtainOptimalLegalCoordiX(optimal_x, optimal_line));
      } else {
        dest_cluster->set_min_x(optimal_line.second);
      }
    } else {
      dest_cluster->set_min_x(obtainOptimalLegalCoordiX(optimal_line.first, optimal_line));
    }

    collapseClusters(front_cluster, dest_cluster);
  }
}

int32_t RowOpt::obtainOptimalLegalCoordiX(int32_t optimal_x, std::pair<int32_t, int32_t>& optimal_line)
{
  int32_t min_x = (optimal_x / _site_width) * _site_width;

  if (min_x < optimal_line.first) {
    min_x += _site_width * 1;
  }

  return min_x;
}

}  // namespace ipl