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
#include "DPOperator.hh"

#include "module/evaluator/wirelength/HPWirelength.hh"

namespace ipl {

DPOperator::DPOperator()
{
}

DPOperator::~DPOperator()
{
  delete _topo_manager;
  delete _grid_manager;
}

void DPOperator::initDPOperator(DPDatabase* database, DPConfig* config)
{
  _database = database;
  _config = config;
  initTopoManager();
  initGridManager();
}

std::pair<int32_t, int32_t> DPOperator::obtainOptimalXCoordiLine(DPInstance* inst)
{
  if (inst->get_pin_list().size() == 0) {
    int32_t inst_x = inst->get_coordi().get_x();
    return std::make_pair(inst_x, inst_x);
  }

  std::vector<int32_t> bound_list;
  int32_t core_max_x = _database->get_layout()->get_max_x();

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

      int32_t bound_lx = bound_pin->get_x_coordi() - pin_offset.first - (inst->get_master()->get_width() / 2);
      if (bound_lx < 0) {
        bound_lx = 0;
      }
      if (bound_lx + inst->get_shape().get_width() > core_max_x) {
        bound_lx = core_max_x - inst->get_shape().get_width();
      }
      bound_list.push_back(bound_lx);
    }
  }
  if (bound_list.empty()) {
    int32_t inst_x = inst->get_coordi().get_x();
    return std::make_pair(inst_x, inst_x);
  }

  std::sort(bound_list.begin(), bound_list.end());
  int32_t center_index = (bound_list.size() - 1) / 2;
  int32_t bound_min_x = bound_list[center_index];
  if (bound_list.size() % 2 == 0) {
    int32_t bound_max_x = bound_list[center_index + 1];
    return std::make_pair(bound_min_x, bound_max_x);
  } else {
    return std::make_pair(bound_min_x, bound_min_x);
  }
}

std::pair<int32_t, int32_t> DPOperator::obtainOptimalYCoordiLine(DPInstance* inst)
{
  if (inst->get_pin_list().size() == 0) {
    int32_t inst_y = inst->get_coordi().get_y();
    return std::make_pair(inst_y, inst_y);
  }

  std::vector<int32_t> bound_list;
  int32_t core_max_y = _database->get_layout()->get_max_y();

  for (auto* inst_pin : inst->get_pin_list()) {
    std::pair<int32_t, int32_t> pin_offset = std::move(inst->calInstPinModifyOffest(inst_pin));

    auto* pin_net = inst_pin->get_net();
    DPPin* up_pin = nullptr;
    DPPin* down_pin = nullptr;
    int32_t max_y = INT32_MIN;
    int32_t min_y = INT32_MAX;
    for (auto* outpin : pin_net->get_pins()) {
      if (outpin == inst_pin) {
        continue;
      }
      if (outpin->get_y_coordi() < min_y) {
        down_pin = outpin;
        min_y = outpin->get_y_coordi();
      }
      if (outpin->get_y_coordi() > max_y) {
        up_pin = outpin;
        max_y = outpin->get_y_coordi();
      }
    }
    // up_pin and down_pin are both bounds
    std::set<DPPin*> bound_pins{up_pin, down_pin};
    for (auto* bound_pin : bound_pins) {
      if (!bound_pin) {
        continue;
      }

      int32_t bound_ly = bound_pin->get_y_coordi() - pin_offset.second - (inst->get_master()->get_height() / 2);
      if (bound_ly < 0) {
        bound_ly = 0;
      }
      if (bound_ly + inst->get_shape().get_height() > core_max_y) {
        bound_ly = core_max_y - inst->get_shape().get_height();
      }
      bound_list.push_back(bound_ly);
    }
  }
  if (bound_list.empty()) {
    int32_t inst_y = inst->get_coordi().get_y();
    return std::make_pair(inst_y, inst_y);
  }

  std::sort(bound_list.begin(), bound_list.end());
  int32_t center_index = (bound_list.size() - 1) / 2;
  int32_t bound_min_y = bound_list[center_index];
  if (bound_list.size() % 2 == 0) {
    int32_t bound_max_y = bound_list[center_index + 1];
    return std::make_pair(bound_min_y, bound_max_y);
  } else {
    return std::make_pair(bound_min_y, bound_min_y);
  }
}

Rectangle<int32_t> DPOperator::obtainOptimalCoordiRegion(DPInstance* inst)
{
  std::pair<int32_t, int32_t> optimal_x_range = std::move(obtainOptimalXCoordiLine(inst));
  std::pair<int32_t, int32_t> optimal_y_range = std::move(obtainOptimalYCoordiLine(inst));

  return Rectangle<int32_t>(optimal_x_range.first, optimal_y_range.first, optimal_x_range.second, optimal_y_range.second);
}

int64_t DPOperator::calInstAffectiveHPWL(DPInstance* inst)
{
  int64_t affective_hpwl = 0;
  for (auto* pin : inst->get_pin_list()) {
    auto* pin_net = pin->get_net();
    affective_hpwl += pin_net->calCurrentHPWL();
  }
  return affective_hpwl;
}

int64_t DPOperator::calInstPairAffectiveHPWL(DPInstance* inst_1, DPInstance* inst_2)
{
  int64_t affective_hpwl = 0;
  std::set<DPNet*> net_set;
  for (auto* pin : inst_1->get_pin_list()) {
    net_set.emplace(pin->get_net());
  }
  for (auto* pin : inst_2->get_pin_list()) {
    net_set.emplace(pin->get_net());
  }
  for (auto* net : net_set) {
    affective_hpwl += net->calCurrentHPWL();
  }

  return affective_hpwl;
}

bool DPOperator::checkIfClustered()
{
  bool flag = true;
  // for (auto* inst : _database->get_design()->get_inst_list()) {
  //   if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
  //     continue;
  //   }

  //   if (!inst->get_belong_cluster()) {
  //     flag = false;
  //     break;
  //   }
  // }
  return flag;
}

void DPOperator::updateInstClustering()
{
  // reset cluster
  _database->get_layout()->resetAllInterval();
  _database->get_design()->clearClusterInfo();

  std::vector<DPInstance*> movable_inst_list;
  pickAndSortMovableInstList(movable_inst_list);

  const std::vector<std::vector<DPInterval*>>& interval_list = _database->get_layout()->get_interval_2d_list();
  int32_t row_height = _database->get_layout()->get_row_height();

  std::map<DPInterval*, DPCluster*> interval_last_cluster;
  for (auto* inst : movable_inst_list) {
    int32_t inst_lx = inst->get_shape().get_ll_x();
    int32_t inst_ux = inst->get_shape().get_ur_x();
    int32_t row_index = inst->get_coordi().get_y() / row_height;
    for (auto* interval : interval_list[row_index]) {
      if (interval->checkInLine(inst_lx, inst_ux)) {
        DPCluster* front_cluster = nullptr;
        auto it = interval_last_cluster.find(interval);
        if (it != interval_last_cluster.end()) {
          front_cluster = it->second;
        }

        if (front_cluster) {
          // collapse inst to cluster
          if (front_cluster->get_max_x() == inst_lx) {
            front_cluster->add_inst(inst);
            inst->set_belong_cluster(front_cluster);
            inst->set_internal_id(front_cluster->get_inst_list().size() - 1);
          } else {
            DPCluster* cluster = createClsuter(inst, interval);
            cluster->set_min_x(inst_lx);
            interval_last_cluster.emplace(interval, cluster);
          }
        } else {
          DPCluster* cluster = createClsuter(inst, interval);
          cluster->set_min_x(inst_lx);
          interval->set_cluster_root(cluster);
          interval_last_cluster.emplace(interval, cluster);
        }

        interval->updateRemainLength(-(inst_ux - inst_lx));
        break;
      }
    }
  }
}

void DPOperator::pickAndSortMovableInstList(std::vector<DPInstance*>& movable_inst_list)
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

DPCluster* DPOperator::createClsuter(DPInstance* inst, DPInterval* interval)
{
  DPCluster* cluster = new DPCluster(inst->get_name());
  cluster->set_belong_interval(interval);
  cluster->set_boundary_min_x(interval->get_min_x());
  cluster->set_boundary_max_x(interval->get_max_x());
  cluster->add_inst(inst);
  inst->set_internal_id(0);
  inst->set_belong_cluster(cluster);
  _database->get_design()->add_cluster(cluster);

  return cluster;
}

bool DPOperator::checkOverlap(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max)
{
  auto overlap_range = std::move(obtainOverlapRange(boundary_min, boundary_max, query_min, query_max));
  if (overlap_range.first > overlap_range.second) {
    return false;
  } else {
    return true;
  }
}

bool DPOperator::checkInNest(Rectangle<int32_t>& inner_box, Rectangle<int32_t>& outer_box)
{
  if (inner_box.get_ll_x() >= outer_box.get_ll_x() && inner_box.get_ur_x() <= outer_box.get_ur_x()
      && inner_box.get_ll_y() >= outer_box.get_ll_y() && inner_box.get_ur_y() <= outer_box.get_ur_y()) {
    return true;
  }
  return false;
}

Rectangle<int32_t> DPOperator::obtainOverlapRectangle(Rectangle<int32_t>& box_1, Rectangle<int32_t>& box_2)
{
  int32_t llx = box_1.get_ll_x() < box_2.get_ll_x() ? box_2.get_ll_x() : box_1.get_ll_x();
  int32_t lly = box_1.get_ll_y() < box_2.get_ll_y() ? box_2.get_ll_y() : box_1.get_ll_y();
  int32_t urx = box_1.get_ur_x() > box_2.get_ur_x() ? box_2.get_ur_x() : box_1.get_ur_x();
  int32_t ury = box_1.get_ur_y() > box_2.get_ur_y() ? box_2.get_ur_y() : box_1.get_ur_y();

  if (llx >= urx || lly >= ury) {
    return Rectangle<int32_t>(0, 0, 0, 0);
  }

  return Rectangle<int32_t>(llx, lly, urx, ury);
}

std::pair<int32_t, int32_t> DPOperator::obtainOverlapRange(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max)
{
  int32_t range_min = boundary_min < query_min ? query_min : boundary_min;
  int32_t range_max = boundary_max > query_max ? query_max : boundary_max;

  return std::make_pair(range_min, range_max);
}

bool DPOperator::checkInBox(int32_t boundary_min, int32_t boundary_max, int32_t query_min, int32_t query_max)
{
  return (query_min >= boundary_min && query_max <= boundary_max);
}

void DPOperator::initTopoManager()
{
  _topo_manager = new TopologyManager();
  auto* design = _database->get_design();
  for (auto* pin : design->get_pin_list()) {
    Node* node = new Node(pin->get_name());
    node->set_location(Point<int32_t>(pin->get_x_coordi(), pin->get_y_coordi()));
    _topo_manager->add_node(node);
  }

  for (auto* net : design->get_net_list()) {
    NetWork* network = new NetWork(net->get_name());
    network->set_net_weight(net->get_netwight());
    DPPin* driver_pin = net->get_driver_pin();
    if (driver_pin) {
      Node* transmitter = _topo_manager->findNodeById(driver_pin->get_pin_id());
      transmitter->set_network(network);
      network->set_transmitter(transmitter);
    }
    for (auto* load_pin : net->get_pins()) {
      if (load_pin == driver_pin) {
        continue;
        ;
      }
      Node* receiver = _topo_manager->findNodeById(load_pin->get_pin_id());
      receiver->set_network(network);
      network->add_receiver(receiver);
    }
    _topo_manager->add_network(network);
  }

  for (auto* inst : design->get_inst_list()) {
    Group* group = new Group(inst->get_name());
    for (auto* pin : inst->get_pin_list()) {
      Node* node = _topo_manager->findNodeById(pin->get_pin_id());
      node->set_group(group);
      group->add_node(node);
    }
    _topo_manager->add_group(group);
  }
}

void DPOperator::initGridManager()
{
  if(_config->isEnableNetworkflow()){
    int beta = 9;
    const int32_t length = static_cast<int32_t> (beta * _database->get_layout()->get_row_height());
    int32_t bin_size_x = length;
    int32_t bin_size_y = length;
    int32_t num_cols = std::ceil(_database->get_layout()->get_max_x() / static_cast<float>(bin_size_x));
    int32_t num_rows = std::ceil(_database->get_layout()->get_max_y() / static_cast<float>(bin_size_y));
    _grid_manager = new GridManager(Rectangle<int32_t>(0, 0, _database->get_layout()->get_max_x(), _database->get_layout()->get_max_y()), 
                                    num_cols, num_rows, bin_size_x, bin_size_y, 1.0, 1);
    updateGridManager();
    initPlaceableArea();

  }else{
    _grid_manager = new GridManager(Rectangle<int32_t>(0, 0, _database->get_layout()->get_max_x(), _database->get_layout()->get_max_y()), 128,
                                    128, 1.0, 1);
  }
  initGridManagerFixedArea();
}


void DPOperator::initPlaceableArea()
{
  auto layout = _database->get_layout();
  Rectangle<int32_t> die_bound(0, 0, layout->get_max_x(), layout->get_max_y());

	for (int32_t i = 0; i < layout->get_row_num(); i++) {
		for (auto* row : layout->get_row_2d_list().at(i)) {
      auto bound = row->get_bound();
      Rectangle<int32_t> bounds32(static_cast<int32_t>(bound.get_ll_x()), 
                                  static_cast<int32_t>(bound.get_ll_y()), 
                                  static_cast<int32_t>(bound.get_ur_x()), 
                                  static_cast<int32_t>(bound.get_ur_y()));
      Rectangle<int32_t> row_overlap = die_bound.get_intersetion(bounds32);
      std::vector<Grid*> overlap_grid_list;
      _grid_manager->obtainOverlapGridList(overlap_grid_list, row_overlap);
      for (auto* grid : overlap_grid_list) {
        int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, row_overlap);
        grid->placeable_area += overlap_area;
      }
		}
	}
}


void DPOperator::initGridManagerFixedArea()
{
  if (!_grid_manager) {
    LOG_WARNING << "grid manager has not been initialized ! ";
    return;
  }

  for (auto* inst : _database->get_design()->get_inst_list()) {
    if (!(inst->get_state() == DPINSTANCE_STATE::kFixed)) {
      continue;
    }

    if ((inst->get_state() == DPINSTANCE_STATE::kFixed) && !isCoreOverlap(inst)) {
      continue;
    }

    // add fix insts.
    std::vector<Grid*> overlap_grid_list;
    auto inst_shape = std::move(inst->get_shape());
    cutOutShape(inst_shape);
    _grid_manager->obtainOverlapGridList(overlap_grid_list, inst_shape);
    for (auto* grid : overlap_grid_list) {
      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, inst->get_shape());
      grid->fixed_area += overlap_area;

      // grid->add_fixed_area(overlap_area);
    }
  }

  // add blockage.
  auto region_list = _database->get_layout()->get_region_list();
  for (auto* region : region_list) {
    if (region->get_type() == DPREGION_TYPE::kFence) {
      std::vector<Grid*> overlap_grid_list;
      for (auto& boundary : region->get_shape_list()) {
        auto boundary_shape = boundary;
        _grid_manager->obtainOverlapGridList(overlap_grid_list, boundary_shape);
        for (auto* grid : overlap_grid_list) {
          // tmp fix overlap area between fixed inst and blockage.
          if (grid->fixed_area != 0) {
            continue;
          }

          // if (grid->get_fixed_area() != 0) {
          //   continue;
          // }
          int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, boundary);

          grid->fixed_area += overlap_area;
          // grid->add_fixed_area(overlap_area);
        }
      }
    }
  }
}

void DPOperator::updateTopoManager()
{
  for (auto* pin : _database->get_design()->get_pin_list()) {
    auto* node = _topo_manager->findNodeById(pin->get_pin_id());
    node->set_location(Point<int32_t>(pin->get_x_coordi(), pin->get_y_coordi()));
  }
}

void DPOperator::updateGridManager()
{
  _grid_manager->clearAllOccupiedArea();
  _grid_manager->clearAllOccupiedNodeNum();

  for (auto* inst : _database->get_design()->get_inst_list()) {
    if (inst->get_coordi().isUnLegal()) {
      continue;
    }
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }

    auto inst_shape = std::move(inst->get_shape());
    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, inst_shape);
    for (auto* grid : overlap_grid_list) {
      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, inst_shape);
      grid->num_node += 1;
      grid->occupied_area += overlap_area;
    }
  }
}

bool DPOperator::isCoreOverlap(DPInstance* inst)
{
  auto inst_shape = std::move(inst->get_shape());
  int32_t max_x = _database->get_layout()->get_max_x();
  int32_t max_y = _database->get_layout()->get_max_y();

  if (inst_shape.get_ur_x() <= 0 || inst_shape.get_ll_y() >= max_y || inst_shape.get_ll_x() >= max_x || inst_shape.get_ur_y() <= 0) {
    return false;
  } else {
    return true;
  }
}

void DPOperator::cutOutShape(Rectangle<int32_t>& shape)
{
  int32_t max_x = _database->get_layout()->get_max_x();
  int32_t max_y = _database->get_layout()->get_max_y();

  int32_t llx = shape.get_ll_x();
  int32_t urx = shape.get_ur_x();
  int32_t lly = shape.get_ll_y();
  int32_t ury = shape.get_ur_y();

  llx < 0 ? llx = 0 : llx;
  lly < 0 ? lly = 0 : lly;
  urx > max_x ? urx = max_x : urx;
  ury > max_y ? ury = max_y : ury;

  shape.set_rectangle(llx, lly, urx, ury);
}

int64_t DPOperator::calTotalHPWL()
{
  HPWirelength hpwl_eval(_topo_manager);
  return hpwl_eval.obtainTotalWirelength() + _database->get_outside_wl();
}

}  // namespace ipl