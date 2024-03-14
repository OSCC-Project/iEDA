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

#include "PostGP.hh"

#include <deque>
#include <tuple>

#include "Legalizer.hh"
#include "log/Log.hh"
#include "utility/Utility.hh"

namespace ipl {

  PostGP::PostGP(Config* pl_config, PlacerDB* placer_db)
  {
    _config = pl_config->get_post_gp_config();

    // create database.
    _database = PostGPDatabase();
    _database._placer_db = placer_db;

    // The equality of topo_manager and placer_db should promise by outside.
    auto* topo_manager = placer_db->get_topo_manager();
    _database._inst_list = placer_db->get_design()->get_instance_list();
    _database._net_list = placer_db->get_design()->get_net_list();
    _database._pin_list = placer_db->get_design()->get_pin_list();
    _database._group_list = topo_manager->get_group_copy_list();
    _database._network_list = topo_manager->get_network_copy_list();
    _database._node_list = topo_manager->get_node_copy_list();
    _database._topo_manager = topo_manager;

    // initialize timing annotation and steiner wl
    _timing_annotation = new TimingAnnotation(topo_manager);
    _steiner_wl = _timing_annotation->get_stwl_ptr();
  }

  PostGP::~PostGP() {
    delete _timing_annotation;
  }

  void PostGP::runIncrTimingPlace() {
    // Record incremental improvement and set loop
    int max_iter = 1;
    float expect_improve_ratio = 0.01;
    float prev_tns, cur_tns;

    for (int i = 0; i < max_iter; i++) {
      prev_tns = _timing_annotation->get_late_tns();

      _timing_annotation->updateCriticalityAndCentralityFull();
      runBufferBalancing();
      _timing_annotation->updateSTATimingFull();
      _timing_annotation->updateCriticalityAndCentralityFull();

      runCellBalancing();
      _timing_annotation->updateSTATimingFull();
      _timing_annotation->updateCriticalityAndCentralityFull();

      runLoadReduction();
      _timing_annotation->updateSTATimingFull();
      _timing_annotation->updateCriticalityAndCentralityFull();

      // _timing_annotation->updateSTATimingFull();
      cur_tns = _timing_annotation->get_late_tns();
      
      bool flag_1 = cur_tns < prev_tns;
      bool flag_2 = (cur_tns - prev_tns) / cur_tns < expect_improve_ratio;
      if(flag_1 || !flag_2){
        break;
      }
    }


    PlacerDBInst.updateTopoManager();
    PlacerDBInst.updateGridManager();
  }

  void PostGP::runBufferBalancing()
  {
    LOG_INFO << "Start buffer balancing...";

    // find single buffer chains
    std::deque<std::tuple<float, Instance*>> buffers;

    for (auto* inst : _database._inst_list) {
      auto* group = _database._group_list[inst->get_inst_id()];
      std::vector<Node*> input_nodes = std::move(group->obtainInputNodes());
      std::vector<Node*> output_nodes = std::move(group->obtainOutputNodes());

      if (input_nodes.size() != 1 || output_nodes.size() != 1) {
        continue;
      }

      Node* in_node = input_nodes[0];
      Node* out_node = output_nodes[0];

      NetWork* in_net = in_node->get_network();
      NetWork* out_net = out_node->get_network();

      if (!in_net || in_net->get_node_list().size() != 2) {
        continue;
      }
      if (!out_net || out_net->get_node_list().size() != 2) {
        continue;
      }

      float criticality = _timing_annotation->get_group_criticality(group);
      if (!Utility().isFloatApproximatelyZero(criticality)) {
        buffers.push_back(std::make_tuple(criticality, inst));
      }
    }

    // sort buffer by criticality
    std::sort(buffers.begin(), buffers.end());

    int32_t moved_buffers = 0;
    int32_t failed = 0;

    int num_buffers = buffers.size();

    for (int32_t i = num_buffers - 1; i >= 0; i--) {
      auto* buffer = std::get<1>(buffers[i]);
      if (!doBufferBalancing(buffer)) {
        failed++;
      }
      moved_buffers++;
    }

    LOG_INFO << "End buffer balancing...";
  }

  void PostGP::runCellBalancing()
  {
    LOG_INFO << "Start cell balancing...";

    // find single buffer chains
    std::deque<std::tuple<float, Instance*>> cells;

    for (auto* inst : _database._inst_list) {
      auto* group = _database._group_list[inst->get_inst_id()];

      float criticality = _timing_annotation->get_group_criticality(group);
      if (!Utility().isFloatApproximatelyZero(criticality)) {
        cells.push_back(std::make_tuple(criticality, inst));
      }
    }

    // sort buffer by criticality
    std::sort(cells.begin(), cells.end());

    int32_t moved_cells = 0;
    int32_t failed = 0;

    int num_cells = cells.size();

    for (int32_t i = num_cells - 1; i >= 0; i--) {
      auto* cell = std::get<1>(cells[i]);
      if (!doCellBalancing(cell)) {
        failed++;
      }
      moved_cells++;
    }

    LOG_INFO << "End cell balancing...";
  }

  bool PostGP::doBufferBalancing(Instance* buffer)
  {
    if (buffer->isFixed()) {
      return false;
    }

    auto inpins = std::move(buffer->get_inpins());
    auto outpins = std::move(buffer->get_outpins());

    Pin* inpin = inpins[0];
    Pin* outpin = outpins[0];

    Net* in_net = inpin->get_net();
    Net* out_net = outpin->get_net();

    Pin* driver = in_net->get_driver_pin();
    if (driver->isIOPort()) {
      return false;
    }

    Pin* sink = out_net->get_sink_pins()[0];

    Node* driver_node = _database._node_list[driver->get_pin_id()];
    Node* out_node = _database._node_list[outpin->get_pin_id()];
    Node* sink_node = _database._node_list[sink->get_pin_id()];
    Node* in_node = _database._node_list[inpin->get_pin_id()];

    float C_w = _timing_annotation->getAvgWireCapPerUnitLength();
    float R_w = _timing_annotation->getAvgWireResPerUnitLength();

    int32_t dbu = _database._placer_db->get_layout()->get_database_unit();
    C_w /= dbu;
    R_w /= dbu;

    float R_0 = _timing_annotation->getOutNodeRes(driver_node);
    float R_1 = _timing_annotation->getOutNodeRes(out_node);

    float C_1 = _timing_annotation->getNodeInputCap(in_node);
    float C_2 = _timing_annotation->getNodeInputCap(sink_node);

    const Point<int32_t> driver_pos = driver_node->get_location();
    const Point<int32_t> sink_pos = sink_node->get_location();
    const float d = Utility().calManhattanDistance(driver_pos, sink_pos);
    const float a = 0;  // Utility().calManhattanDistance(in_pin_pos, out_pin_pos);

    if (Utility().isFloatApproximatelyZero(d)) {
      return false;
    }

    float d_1
      = std::min(d - a, std::max(0.0f, (C_w * R_1 - C_w * R_0 + R_w * C_2 - R_w * C_1 + (C_w * d - a * C_w) * R_w) / (2 * C_w * R_w)));

    float dx = sink_pos.get_x() - driver_pos.get_x();
    float dy = sink_pos.get_y() - driver_pos.get_y();
    float scaling = d_1 / d;

    float px = scaling * dx + driver_pos.get_x();
    float py = scaling * dy + driver_pos.get_y();

    // calculate the cost and legal the inst.
    float old_cost = this->calCurrentCost(buffer);

    // printTimingInfoForSTADebug(buffer);
    this->runIncrLGAndUpdateTiming(buffer, px, py);
    // printTimingInfoForSTADebug(buffer);

    float new_cost = this->calCurrentCost(buffer);

    if (new_cost > old_cost || Utility().isFloatPairApproximatelyEqual(new_cost, old_cost)) {
      // rollback
      bool rollback_flag = this->runRollback(buffer, false);
      if (!rollback_flag) {
        LOG_INFO << "Cannot rollback to legalized position!!!";
        exit(1);
      }
      return false;
    }
    else {
      // clear rollback stack
      bool rollback_flag = this->runRollback(buffer, true);
      return rollback_flag;
    }
  }

  bool PostGP::doCellBalancing(Instance* inst)
  {
    if (inst->isFixed()) {
      return false;
    }

    auto inpins = std::move(inst->get_inpins());
    auto outpins = std::move(inst->get_outpins());

    float avg_px = 0;
    float avg_py = 0;
    float total_weight = 0;
    int32_t num_pos = 0;

    for (auto* outpin : outpins) {
      Net* out_net = outpin->get_net();
      for (auto* inpin : inpins) {
        Net* in_net = inpin->get_net();

        Pin* driver = in_net->get_driver_pin();
        if (driver->isIOPort()) {
          return false;
        }

        float C_w = _timing_annotation->getAvgWireCapPerUnitLength();
        float R_w = _timing_annotation->getAvgWireResPerUnitLength();

        int32_t dbu = _database._placer_db->get_layout()->get_database_unit();
        C_w /= dbu;
        R_w /= dbu;

        Node* driver_node = _database._node_list[driver->get_pin_id()];
        Node* out_node = _database._node_list[outpin->get_pin_id()];
        Node* in_node = _database._node_list[inpin->get_pin_id()];

        float R_0 = _timing_annotation->getOutNodeRes(driver_node);
        float R_1 = _timing_annotation->getOutNodeRes(out_node);

        for (auto* sink : out_net->get_sink_pins()) {
          Node* sink_node = _database._node_list[sink->get_pin_id()];
          float C_1 = _timing_annotation->getNodeInputCap(in_node);
          float C_2 = _timing_annotation->getNodeInputCap(sink_node);

          const Point<int32_t> driver_pos = driver_node->get_location();
          const Point<int32_t> sink_pos = sink_node->get_location();
          const float d = Utility().calManhattanDistance(driver_pos, sink_pos);
          const float a = 0;  // Utility().calManhattanDistance(in_pin_pos, out_pin_pos);

          if (Utility().isFloatApproximatelyZero(d)) {
            continue;
          }

          float w_0 = _timing_annotation->get_node_importance(driver_node);
          float w_1 = _timing_annotation->get_node_importance(sink_node);

          if (Utility().isFloatApproximatelyZero(w_0 + w_1)) {
            continue;
          }

          float actual_D = (C_w * w_1 * R_1 - C_w * w_0 * R_0 + R_w * w_1 * C_2 - R_w * w_0 * C_1 + (C_w * d - a * C_w) * R_w * w_1)
            / (C_w * R_w * w_1 + C_w * R_w * w_0);

          float d_0 = std::min(d - a, std::max(0.0f, actual_D));

          float dx = sink_pos.get_x() - driver_pos.get_x();
          float dy = sink_pos.get_y() - driver_pos.get_y();
          float scaling = d_0 / d;

          float weight = w_0 + w_1;

          avg_px += weight * (scaling * dx + driver_pos.get_x());
          avg_py += weight * (scaling * dy + driver_pos.get_y());
          total_weight += weight;
          num_pos++;
        }
      }
    }

    if (num_pos > 0 && total_weight > 0) {
      avg_px /= total_weight;
      avg_py /= total_weight;

      // calculate the cost and legal the inst.
      float old_cost = this->calCurrentCost(inst);

      // printTimingInfoForSTADebug(inst);
      this->runIncrLGAndUpdateTiming(inst, avg_px, avg_py);
      // printTimingInfoForSTADebug(inst);
      
      float new_cost = this->calCurrentCost(inst);

      if (new_cost > old_cost || Utility().isFloatPairApproximatelyEqual(new_cost, old_cost)) {
        // rollback
        bool rollback_flag = this->runRollback(inst, false);
        if (!rollback_flag) {
          LOG_INFO << "Cannot rollback to legalized position!!!";
          exit(1);
        }
        return false;
      }
      else {
        bool rollback_flag = this->runRollback(inst, true);
        return rollback_flag;
      }
    }
    else {
      return false;
    }
  }

  float PostGP::calCurrentCost(Instance* inst)
  {
    float cost = 0;
    for (auto* pin : inst->get_pins()) {
      auto* pin_net = pin->get_net();
      if (pin_net) {
        for (auto* net_sink_pin : pin_net->get_sink_pins()) {
          if (net_sink_pin->get_instance() == inst) {
            continue;
          }

          auto* node = _database._node_list[net_sink_pin->get_pin_id()];
          float node_importance = _timing_annotation->get_node_importance(node);
          float late_at = _timing_annotation->get_node_late_arrival_time(node->get_node_id());

          cost += node_importance * late_at;
        }
      }
    }
    return cost;
  }

  bool PostGP::runIncrLGAndUpdateTiming(Instance* inst, int32_t x, int32_t y)
  {
    inst->update_coordi(x, y);
    LegalizerInst.updateInstance(inst);
    LegalizerInst.runIncrLegalize();  // Need to compute cost, so not rollback before calCost.

    std::vector<NetWork*> influenced_networks;
    for (auto* pin : inst->get_pins()) {
      Node* node = _database._node_list[pin->get_pin_id()];
      influenced_networks.push_back(node->get_network());
    }

    _steiner_wl->updatePartOfNetWorkPointPair(influenced_networks);
    std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map;
    for (auto* network : influenced_networks) {
      const auto& point_pair_list = _steiner_wl->obtainPointPairList(network);
      net_id_to_points_map.emplace(network->get_network_id(), point_pair_list);
    }

    // need to modify after, sta should support side cell update.
    std::vector<std::string> front_insts_name = obtainFrontInstNameList(inst);

    iPLAPIInst.updateTimingInstMovement(_database._topo_manager, net_id_to_points_map, front_insts_name);
    // iPLAPIInst.updateTimingInstMovement(_database._topo_manager, net_id_to_points_map, std::vector<std::string>{inst->get_name()});

    return true;
  }

  bool PostGP::runRollback(Instance* inst, bool clear_but_not_rollback)
  {
    bool flag = LegalizerInst.runRollback(clear_but_not_rollback);

    if (clear_but_not_rollback) {
      return true;
    }

    std::vector<NetWork*> influenced_networks;
    for (auto* pin : inst->get_pins()) {
      Node* node = _database._node_list[pin->get_pin_id()];
      influenced_networks.push_back(node->get_network());
    }

    _steiner_wl->updatePartOfNetWorkPointPair(influenced_networks);
    std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map;
    for (auto* network : influenced_networks) {
      const auto& point_pair_list = _steiner_wl->obtainPointPairList(network);
      net_id_to_points_map.emplace(network->get_network_id(), point_pair_list);
    }

    // need to modify after, sta should support side cell update.
    std::vector<std::string> front_insts_name = obtainFrontInstNameList(inst);

    iPLAPIInst.updateTimingInstMovement(_database._topo_manager, net_id_to_points_map, front_insts_name);
    // iPLAPIInst.updateTimingInstMovement(_database._topo_manager, net_id_to_points_map, std::vector<std::string>{inst->get_name()});

    return flag;
  }

  std::vector<std::string> PostGP::obtainFrontInstNameList(Instance* inst){
   std::vector<std::string> name_list;
   std::vector<Pin*> input_pin = inst->get_inpins();
   for(auto* pin : input_pin){
    auto* pin_net = pin->get_net();
    if(pin_net){
      auto* driver = pin_net->get_driver_pin();
      if(driver){
        auto* front_inst = driver->get_instance();
        if(front_inst){
          name_list.push_back(front_inst->get_name());
        }
      }
    }
   }
   if(name_list.empty()){
    name_list.push_back(inst->get_name());
   }
   
   return name_list;
  }

  void PostGP::runLoadReduction()
  {
    LOG_INFO << "Start load optimization";

    std::map<Instance*, bool> visited;
    for (auto* inst : _database._inst_list) {
      visited.emplace(inst, false);
    }

    float late_threshold = -50.0f;
    int32_t counter_moved = 0;
    int32_t couter_failed = 0;

    std::deque<std::tuple<float, Net*>> ordered_nets;
    for (auto* net : _database._net_list) {
      auto* network = _database._network_list[net->get_net_id()];

      float criticality = _timing_annotation->get_network_criticality(network);
      ordered_nets.push_back(std::make_tuple(criticality, net));
    }
    std::sort(ordered_nets.begin(), ordered_nets.end());

    int32_t num_nets = ordered_nets.size();

    for (int32_t i = num_nets - 1; i >= 0; i--) {
      auto* net = std::get<1>(ordered_nets[i]);

      auto* driver = net->get_driver_pin();
      auto* driver_node = _database._node_list[driver->get_pin_id()];

      if (!driver || _timing_annotation->get_node_late_slack(driver_node->get_node_id()) >= 0) {
        continue;
      }

      for (auto* sink : net->get_sink_pins()) {
        auto* sink_inst = sink->get_instance();

        if (sink->isIOPort() || !sink_inst) {
          continue;
        }

        if (visited[sink_inst]) {
          continue;
        }
        else {
          visited[sink_inst] = true;
        }

        auto* sink_cell = sink_inst->get_cell_master();
        if (sink_inst->isFixed() || sink_cell->isFlipflop() || sink_cell->isClockBuffer()) {
          continue;
        }

        bool is_critical = false;
        auto inst_pos = sink_inst->get_coordi();
        for (auto* out_pin : sink_inst->get_outpins()) {
          if (_timing_annotation->get_node_late_slack(out_pin->get_pin_id()) < late_threshold) {
            is_critical = true;
            break;
          }
        }

        if (!is_critical) {
          auto driver_pos = driver->get_center_coordi();

          float driver_x = driver_pos.get_x();
          float driver_y = driver_pos.get_y();

          // calculate the cost and legal the inst.
          float old_cost = this->calCurrentCost(sink_inst);
          this->runIncrLGAndUpdateTiming(sink_inst, driver_x, driver_y);
          float new_cost = this->calCurrentCost(sink_inst);

          if (new_cost > old_cost || Utility().isFloatPairApproximatelyEqual(new_cost, old_cost)) {
            // rollback
            bool rollback_flag = this->runRollback(sink_inst, false);
            if (!rollback_flag) {
              LOG_INFO << "Cannot rollback to legalized position!!!";
              exit(1);
            }
            couter_failed++;
          }
          else {
            this->runRollback(sink_inst, true);
            counter_moved++;
          }
        }
      }
    }
  }

  void PostGP::printTimingInfoForSTADebug(Instance* inst){
    LOG_INFO << std::endl;
    LOG_WARNING << "Debug Instance: " << inst->get_name()
                << " Position: (" << inst->get_coordi().get_x() << ","
                << inst->get_coordi().get_y() << ")";
    
    for(auto* pin : inst->get_pins()){
      auto* pin_net = pin->get_net();
      if(pin_net){
        LOG_INFO << "Pin: " << pin->get_name() << " Associated Net: " << pin_net->get_name();
        for(auto* net_sink_pin : pin_net->get_sink_pins()){
          auto* node = _database._node_list[net_sink_pin->get_pin_id()];
          LOG_INFO << " --- " << "pin: " << node->get_name() << " arrival time: "
                   << _timing_annotation->get_node_late_arrival_time(node->get_node_id());
        }
      }
    }
    LOG_INFO << std::endl;
  }

}  // namespace ipl
