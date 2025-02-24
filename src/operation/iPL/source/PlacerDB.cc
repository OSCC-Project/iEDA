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
 * @Author: S.J Chen
 * @Date: 2022-02-18 11:24:20
 * @LastEditTime: 2023-02-22 11:34:09
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/PlacerDB.cc
 * Contact : https://github.com/sjchanson
 */

#include "PlacerDB.hh"

#include "Log.hh"
#include "data/Rectangle.hh"
#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>

// #include <boost/polygon/geometry.hpp>

namespace gtl = boost::polygon;
using namespace gtl::operators;
typedef gtl::polygon_90_set_data<int> PolygonSet;

namespace ipl {

  PlacerDB& PlacerDB::getInst()
  {
    if (!_s_placer_db_instance) {
      _s_placer_db_instance = new PlacerDB();
    }

    return *_s_placer_db_instance;
  }

  void PlacerDB::destoryInst()
  {
    if (_s_placer_db_instance) {
      delete _s_placer_db_instance;
      _s_placer_db_instance = nullptr;
    }
  }

  PlacerDB::PlacerDB() : _config(nullptr), _db_wrapper(nullptr), _topo_manager(nullptr), _grid_manager(nullptr)
  {
  }

  PlacerDB::~PlacerDB()
  {
    if (_config) {
      delete _config;
      _config = nullptr;
    }
    if (_db_wrapper) {
      delete _db_wrapper;
      _db_wrapper = nullptr;
    }
    if (_topo_manager) {
      delete _topo_manager;
      _topo_manager = nullptr;
    }
    if (_grid_manager) {
      delete _grid_manager;
      _grid_manager = nullptr;
    }
  }

  void PlacerDB::initPlacerDB(std::string pl_json_path, DBWrapper* db_wrapper)
  {
    _db_wrapper = db_wrapper;
    updatePlacerConfig(pl_json_path);
    // sortDataForParallel();
    initTopoManager();
    initGridManager();
    updateTopoManager();
    updateGridManager();
    // updateFromSourceDataBase();
    adaptTargetDensity();
    printPlacerDB();
  }

  void PlacerDB::updatePlacerConfig(std::string pl_json_path)
  {
    if (_config) {
      delete _config;
    }
    _config = new Config(pl_json_path);
  }

  void PlacerDB::initTopoManager()
  {
    if (_topo_manager) {
      delete _topo_manager;
    }

    _topo_manager = new TopologyManager();
    auto* pl_design = this->get_design();
    this->initNodes(pl_design);
    this->initNetworks(pl_design);
    this->initGroups(pl_design);
    this->initArcs();
  }

  void PlacerDB::initNodes(Design* pl_design) {
    for (auto* pin : pl_design->get_pin_list()) {
      Node* node = new Node(pin->get_name());
      node->set_location(std::move(pin->get_center_coordi()));

      // set node type.
      if (pin->isInstanceInput() || pin->isIOInput()) {
        node->set_node_type(NODE_TYPE::kInput);
      }
      else if (pin->isInstanceOutput() || pin->isIOOutput()) {
        node->set_node_type(NODE_TYPE::kOutput);
      }
      else if (pin->isInstanceInputOutput() || pin->isIOInputOutput()) {
        node->set_node_type(NODE_TYPE::kInputOutput);
      }
      else {
        node->set_node_type(NODE_TYPE::kNone);
      }

      // set is io node
      if (pin->isIOPort()) {
        node->set_is_io();
      }
      _topo_manager->add_node(node);
    }
  }

  void PlacerDB::initNetworks(Design* pl_design) {
    for (auto* net : pl_design->get_net_list()) {
      NetWork* network = new NetWork(net->get_name());

      network->set_net_weight(net->get_net_weight());

      // set network type.
      if (net->isClockNet()) {
        network->set_network_type(NETWORK_TYPE::kClock);
      }
      else if (net->isFakeNet()) {
        network->set_network_type(NETWORK_TYPE::kFakeNet);
      }
      else if (net->isSignalNet()) {
        network->set_network_type(NETWORK_TYPE::kSignal);
      }
      else {
        network->set_network_type(NETWORK_TYPE::kNone);
      }

      Pin* driver_pin = net->get_driver_pin();
      if (driver_pin) {
        Node* transmitter = _topo_manager->findNodeById(driver_pin->get_pin_id());
        transmitter->set_network(network);
        network->set_transmitter(transmitter);
      }

      for (auto* loader_pin : net->get_sink_pins()) {
        Node* receiver = _topo_manager->findNodeById(loader_pin->get_pin_id());
        receiver->set_network(network);
        network->add_receiver(receiver);
      }

      _topo_manager->add_network(network);
    }
  }

  void PlacerDB::initGroups(Design* pl_design) {
    for (auto* inst : pl_design->get_instance_list()) {
      Group* group = new Group(inst->get_name());

      // set group type.
      auto* cell_master = inst->get_cell_master();
      if (cell_master) {
        if (cell_master->isFlipflop()) {
          group->set_group_type(GROUP_TYPE::kFlipflop);
        }
        else if (cell_master->isClockBuffer()) {
          group->set_group_type(GROUP_TYPE::kClockBuffer);
        }
        else if (cell_master->isLogicBuffer()) {
          group->set_group_type(GROUP_TYPE::kLogicBuffer);
        }
        else if (cell_master->isMacro()) {
          group->set_group_type(GROUP_TYPE::kMacro);
        }
        else if (cell_master->isIOCell()) {
          group->set_group_type(GROUP_TYPE::kIOCell);
        }
        else if (cell_master->isLogic()) {
          group->set_group_type(GROUP_TYPE::kLogic);
        }
        else {
          group->set_group_type(GROUP_TYPE::kNone);
        }
      }
      else {
        group->set_group_type(GROUP_TYPE::kNone);
      }


      for (auto* pin : inst->get_pins()) {
        Node* node = _topo_manager->findNodeById(pin->get_pin_id());
        node->set_group(group);
        group->add_node(node);
      }

      _topo_manager->add_group(group);
    }
  }

  void PlacerDB::initArcs() {
    for (auto* node : _topo_manager->get_node_list()) {
      // TODO: Consider the INPUT & OUTPUT Case.
      if (node->is_io_node()) {
        if (node->get_node_type() == NODE_TYPE::kOutput) {
          this->generateNetArc(node);
        }
      }
      else {
        if (node->get_node_type() == NODE_TYPE::kInput) {
          this->generateNetArc(node);
        }
        else if (node->get_node_type() == NODE_TYPE::kOutput) {
          this->generateGroupArc(node);
        }
      }
    }
  }

  void PlacerDB::generateNetArc(Node* node) {
    auto* network = node->get_network();
    if (network) {
      auto* driver_node = network->get_transmitter();
      if (driver_node) {
        Arc* net_arc = new Arc(driver_node, node);
        net_arc->set_arc_type(ARC_TYPE::kNetArc);
        driver_node->add_output_arc(net_arc);
        node->add_input_arc(net_arc);
        _topo_manager->add_arc(net_arc);
      }
    }
  }

  void PlacerDB::generateGroupArc(Node* node) {
    auto* group = node->get_group();
    if (group) {
      auto input_list = group->obtainInputNodes();
      for (auto* input_node : input_list) {
        NetWork* input_net = input_node->get_network();
        if (input_net->get_network_type() != NETWORK_TYPE::kClock && group->get_group_type() == GROUP_TYPE::kFlipflop) {
          continue;
        }

        Arc* group_arc = new Arc(input_node, node);
        group_arc->set_arc_type(ARC_TYPE::kGroupArc);
        input_node->add_output_arc(group_arc);
        node->add_input_arc(group_arc);
        _topo_manager->add_arc(group_arc);
      }
    }
  }

  void PlacerDB::initGridManager()
  {
    Rectangle<int32_t> core_shape = this->get_layout()->get_core_shape();
    int32_t row_height = this->get_layout()->get_row_height();
    int32_t site_width = this->get_layout()->get_site_width();

    _grid_manager = new GridManager(core_shape, core_shape.get_width() / site_width, core_shape.get_height() / row_height, 1.0, 1);
    initGridManagerFixedArea();
  }

  void PlacerDB::initGridManagerFixedArea()
  {
    if (!_grid_manager) {
      LOG_WARNING << "grid manager has not been initialized ! ";
      return;
    }

    for (auto* inst : this->get_design()->get_instance_list()) {
      if (inst->isOutsideInstance()) {
        continue;
      }

      if (!inst->isFixed()) {
        continue;
      }

      // add fix insts.
      std::vector<Grid*> overlap_grid_list;
      auto inst_shape = std::move(inst->get_shape());
      _grid_manager->obtainOverlapGridList(overlap_grid_list, inst_shape);
      for (auto* grid : overlap_grid_list) {
        int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, inst->get_shape());

        grid->fixed_area += overlap_area;
        // grid->add_fixed_area(overlap_area);
      }
    }

    // add blockage.
    auto region_list = this->get_design()->get_region_list();
    for (auto* region : region_list) {
      if (region->isFence()) {
        std::vector<Grid*> overlap_grid_list;
        for (auto boundary : region->get_boundaries()) {
          _grid_manager->obtainOverlapGridList(overlap_grid_list, boundary);
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

  void PlacerDB::sortDataForParallel()
  {
    auto* design = this->get_design();
    design->sortDataForParallel();
    initIgnoreNets(_config->get_ignore_net_degree());
  }

  void PlacerDB::initIgnoreNets(int32_t ignore_net_degree)
  {
    for (auto* net : this->get_design()->get_net_list()) {
      int32_t net_degree = net->get_pins().size();

      if (net_degree < 2 || net_degree > ignore_net_degree) {
        net->set_net_state(NET_STATE::kDontCare);
        net->set_netweight(0.0f);
      }
    }
  }

  void PlacerDB::updateTopoManager()
  {
    for (auto* pin : this->get_design()->get_pin_list()) {
      auto* node = _topo_manager->findNodeById(pin->get_pin_id());
      node->set_location(std::move(pin->get_center_coordi()));
    }
  }

  void PlacerDB::updateGridManager()
  {
    _grid_manager->clearAllOccupiedArea();

    for (auto* inst : this->get_design()->get_instance_list()) {
      if (inst->isOutsideInstance()) {
        continue;
      }

      if (inst->get_coordi().isUnLegal()) {
        continue;
      }

      if (inst->isFixed()) {  // Fixed area has been already add.
        continue;
      }

      auto inst_shape = std::move(inst->get_shape());
      std::vector<Grid*> overlap_grid_list;
      _grid_manager->obtainOverlapGridList(overlap_grid_list, inst_shape);
      for (auto* grid : overlap_grid_list) {
        int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, inst_shape);

        grid->occupied_area += overlap_area;

        // grid->add_area(overlap_area);
      }
    }
  }

  void PlacerDB::updateFromSourceDataBase()
  {
    _db_wrapper->updateFromSourceDataBase();
    initTopoManager();
    updateTopoManager();
    updateGridManager();
  }

  void PlacerDB::updateFromSourceDataBase(std::vector<std::string> inst_list)
  {
    _db_wrapper->updateFromSourceDataBase(inst_list);
  }

  void PlacerDB::updateInstancesForDebug(std::vector<Instance*> inst_list)
  {
    auto* design = this->get_design();
    for (auto* inst : inst_list) {
      design->add_instance(inst);
    }
  }

  void PlacerDB::printPlacerDB() const
  {
    printLayoutInfo();
    printInstanceInfo();
    printNetInfo();
    printPinInfo();
    printRegionInfo();
  }

  void PlacerDB::printLayoutInfo() const
  {
    Design* design = this->get_design();
    const Layout* layout = this->get_layout();
    PolygonSet ps;
    std::string design_name = design->get_design_name();
    int32_t database_unit = layout->get_database_unit();
    Rectangle<int32_t> die_rect = layout->get_die_shape();
    Rectangle<int32_t> core_rect = layout->get_core_shape();
    int32_t row_height = layout->get_row_height();
    int32_t site_width = layout->get_site_width();

    LOG_INFO << "Design name : " << design_name;
    LOG_INFO << "Database unit : " << database_unit;
    LOG_INFO << "Die rectangle : " << die_rect.get_ll_x() << "," << die_rect.get_ll_y() << " " << die_rect.get_ur_x() << ","
      << die_rect.get_ur_y();
    LOG_INFO << "Core rectangle : " << core_rect.get_ll_x() << "," << core_rect.get_ll_y() << " " << core_rect.get_ur_x() << ","
      << core_rect.get_ur_y();
    LOG_INFO << "Row height : " << row_height;
    LOG_INFO << "Site width : " << site_width;

    int64_t core_area = static_cast<int64_t>(core_rect.get_width()) * static_cast<int64_t>(core_rect.get_height());
    int64_t place_instance_area = 0;
    int64_t non_place_instance_area = 0;

    for (auto* inst : design->get_instance_list()) {
      // Ignore the insts outside the core.
      // if (inst->get_cell_master() && inst->get_cell_master()->isIOCell()) {
      //   continue;
      // }
      // for ispd's benchmark
      // if (inst->isOutsideInstance()) {
      //   continue;
      // }

      int64_t inst_width = static_cast<int64_t>(inst->get_shape().get_width());
      int64_t inst_height = static_cast<int64_t>(inst->get_shape().get_height());
      if (inst->isFixed()) {
        ps.insert(gtl::rectangle_data<int>(inst->get_coordi().get_x(), inst->get_coordi().get_y(),
          inst->get_coordi().get_x() + inst_width, inst->get_coordi().get_y() + inst_height));
      }
      else {
        place_instance_area += inst_width * inst_height;
      }
    }
    // PolygonSet ps; // Commented out as it is not used
    
    for (auto* blockage : design->get_region_list()) {
      for (auto boundary : blockage->get_boundaries()) {
        // non_place_instance_area += boundary_width * boundary_height;
        auto rect = gtl::rectangle_data<int>(boundary.get_ll_x(), boundary.get_ll_y(), boundary.get_ur_x(), boundary.get_ur_y());
        ps.insert(rect) ;
        
      }
    }
    ps &= gtl::rectangle_data<int>(core_rect.get_ll_x(), core_rect.get_ll_y(), core_rect.get_ur_x(), core_rect.get_ur_y());

    std::vector<gtl::rectangle_data<int>> rects;
    ps.get_rectangles(rects);
    
    int64_t total_blockage_area = 0;
    for (const auto &rect : rects) {
      total_blockage_area += 1LL* boost::polygon::area(rect);
    }
    // LOG_INFO << "Total blockage area : " << total_blockage_area;
    non_place_instance_area = total_blockage_area;
    LOG_INFO << "Core area : " << core_area;
    LOG_INFO << "Non place instance area : " << non_place_instance_area;
    LOG_INFO << "Place instance area : " << place_instance_area;

    double util = static_cast<double>(place_instance_area) / (core_area - non_place_instance_area) * 100;
    LOG_INFO << "Uitization(%) : " << util;
    LOG_ERROR_IF(util > 100.1) << "Utilization exceeds 100%";
  }

  float PlacerDB::obtainUtilization()
  {
    Design* design = this->get_design();
    const Layout* layout = this->get_layout();
    Rectangle<int32_t> core_rect = layout->get_core_shape();
    int64_t core_area = static_cast<int64_t>(core_rect.get_width()) * static_cast<int64_t>(core_rect.get_height());
    int64_t place_instance_area = 0;
    int64_t non_place_instance_area = 0;
    PolygonSet ps;
    for (auto* inst : design->get_instance_list()) {
      // Ignore the insts outside the core.
      // if (inst->get_cell_master() && inst->get_cell_master()->isIOCell()) {
      //   continue;
      // }
      // for ispd's benchmark
      // if (inst->isOutsideInstance()) {
      //   continue;
      // }

      int64_t inst_width = static_cast<int64_t>(inst->get_shape().get_width());
      int64_t inst_height = static_cast<int64_t>(inst->get_shape().get_height());
      if (inst->isFixed()) {
        ps.insert(gtl::rectangle_data<int>(inst->get_coordi().get_x(), inst->get_coordi().get_y(),
          inst->get_coordi().get_x() + inst_width, inst->get_coordi().get_y() + inst_height));
      }
      else {
        place_instance_area += inst_width * inst_height;
      }
    }
    // PolygonSet ps; // Commented out as it is not used
    
    for (auto* blockage : design->get_region_list()) {
      for (auto boundary : blockage->get_boundaries()) {
        // non_place_instance_area += boundary_width * boundary_height;
        auto rect = gtl::rectangle_data<int>(boundary.get_ll_x(), boundary.get_ll_y(), boundary.get_ur_x(), boundary.get_ur_y());
        ps.insert(rect) ;
        
      }
    }
    ps &= gtl::rectangle_data<int>(core_rect.get_ll_x(), core_rect.get_ll_y(), core_rect.get_ur_x(), core_rect.get_ur_y());

    std::vector<gtl::rectangle_data<int>> rects;
    ps.get_rectangles(rects);
    
    int64_t total_blockage_area = 0;
    for (const auto &rect : rects) {
      total_blockage_area += 1LL* boost::polygon::area(rect);
    }
    // LOG_INFO << "Total blockage area : " << total_blockage_area;
    non_place_instance_area = total_blockage_area;

    float util = static_cast<float>(place_instance_area) / (core_area - non_place_instance_area);
    return util;
  }

  void PlacerDB::adaptTargetDensity()
  {
    float cur_util = this->obtainUtilization();
    float user_target_density = this->get_placer_config()->get_nes_config().get_target_density();
    if (user_target_density < cur_util) {
      float setting_util = cur_util + 0.001;
      this->get_placer_config()->get_nes_config().set_target_density(setting_util);
    }
  }

  void PlacerDB::printInstanceInfo() const
  {
    const Layout* layout = this->get_layout();
    Design* design = this->get_design();

    int32_t num_instances = 0;
    int32_t num_macros = 0;
    int32_t num_logic_insts = 0;
    int32_t num_flipflop_cells = 0;
    int32_t num_clock_buffers = 0;
    int32_t num_logic_buffers = 0;
    int32_t num_io_cells = 0;
    int32_t num_physical_insts = 0;
    int32_t num_outside_insts = 0;
    int32_t num_fake_instances = 0;
    int32_t num_unplaced_instances = 0;
    int32_t num_placed_instances = 0;
    int32_t num_fixed_instances = 0;
    int32_t num_cell_masters = static_cast<int32_t>(layout->get_cell_list().size());

    for (auto* inst : design->get_instance_list()) {
      num_instances++;

      Cell* cell_master = inst->get_cell_master();
      if (cell_master) {
        if (cell_master->isMacro()) {
          num_macros++;
        }
        else if (cell_master->isLogic()) {
          num_logic_insts++;
        }
        else if (cell_master->isFlipflop()) {
          num_flipflop_cells++;
        }
        else if (cell_master->isClockBuffer()) {
          num_clock_buffers++;
        }
        else if (cell_master->isLogicBuffer()) {
          num_logic_buffers++;
        }
        else if (cell_master->isIOCell()) {
          num_io_cells++;
        }
        else if (cell_master->isPhysicalFiller()) {
          num_physical_insts++;
        }
        else {
          LOG_WARNING << "Instance : " + inst->get_name() + " doesn't have a cell type.";
        }
      }

      if (inst->isFakeInstance()) {
        num_fake_instances++;
      }
      else if (inst->isNormalInstance()) {
        //
      }
      else if (inst->isOutsideInstance()) {
        num_outside_insts++;
      }
      else {
        // LOG_WARNING << "Instance : " + inst->get_name() + " doesn't have a instance type.";
      }

      if (inst->isUnPlaced()) {
        num_unplaced_instances++;
      }
      else if (inst->isPlaced()) {
        num_placed_instances++;
      }
      else if (inst->isFixed()) {
        num_fixed_instances++;
      }
      else {
        LOG_WARNING << "Instance : " + inst->get_name() + " doesn't have a instance state.";
      }
    }

    LOG_INFO << "Instances Num : " << num_instances;
    LOG_INFO << "1. Macro Num : " << num_macros;
    LOG_INFO << "2. Stdcell Num : " << num_instances - num_macros;
    LOG_INFO << "2.1 Logic Instances : " << num_logic_insts;
    LOG_INFO << "2.2 Flipflops : " << num_flipflop_cells;
    LOG_INFO << "2.3 Clock Buffers : " << num_clock_buffers;
    LOG_INFO << "2.4 Logic Buffers : " << num_logic_buffers;
    LOG_INFO << "2.5 IO Cells : " << num_io_cells;
    LOG_INFO << "2.6 Physical Instances : " << num_physical_insts;
    LOG_INFO << "Core Outside Instances : " << num_outside_insts;
    LOG_INFO << "Fake Instances : " << num_fake_instances;
    LOG_INFO << "Unplaced Instances Num : " << num_unplaced_instances;
    LOG_INFO << "Placed Instances Num : " << num_placed_instances;
    LOG_INFO << "Fixed Instances Num : " << num_fixed_instances;
    LOG_INFO << "Optional CellMaster Num : " << num_cell_masters;
  }

  void PlacerDB::printNetInfo() const
  {
    Design* design = this->get_design();

    int32_t num_nets = 0;
    int32_t num_clock_nets = 0;
    int32_t num_reset_nets = 0;
    int32_t num_signal_nets = 0;
    int32_t num_fake_nets = 0;
    int32_t num_normal_nets = 0;
    int32_t num_dontcare_nets = 0;

    int32_t num_no_type_nets = 0;
    int32_t num_no_state_nets = 0;

    for (auto* net : design->get_net_list()) {
      num_nets++;
      if (net->isClockNet()) {
        num_clock_nets++;
      }
      else if (net->isResetNet()) {
        num_reset_nets++;
      }
      else if (net->isSignalNet()) {
        num_signal_nets++;
      }
      else if (net->isFakeNet()) {
        num_fake_nets++;
      }
      else {
        num_no_type_nets++;
      }

      if (net->isNormalStateNet()) {
        num_normal_nets++;
      }
      else if (net->isDontCareNet()) {
        num_dontcare_nets++;
      }
      else {
        num_no_state_nets++;
      }
    }

    LOG_INFO << "Nets Num : " << num_nets;
    LOG_INFO << "1. ClockNets Num : " << num_clock_nets;
    LOG_INFO << "2. ResetNets Num : " << num_reset_nets;
    LOG_INFO << "3. SignalNets Num : " << num_signal_nets;
    LOG_INFO << "4. FakeNets Num : " << num_fake_nets;
    LOG_INFO << "Don't Care Net Num : " << num_dontcare_nets;

    if (num_no_type_nets != 0) {
      LOG_WARNING << "Existed Nets don't have NET_TYPE : " << num_no_type_nets;
    }
    if (num_no_state_nets != 0) {
      LOG_WARNING << "Existed Nets don't have NET_STATE : " << num_no_state_nets;
    }
  }

  void PlacerDB::printPinInfo() const
  {
    Design* design = this->get_design();

    int32_t num_pins = 0;
    int32_t num_io_ports = 0;
    int32_t num_instance_ports = 0;
    int32_t num_fake_pins = 0;

    for (auto* pin : design->get_pin_list()) {
      num_pins++;
      if (pin->isIOPort()) {
        num_io_ports++;
      }
      else if (pin->isInstancePort()) {
        num_instance_ports++;
      }
      else if (pin->isFakePin()) {
        num_fake_pins++;
      }
      else {
        LOG_WARNING << "Pin : " + pin->get_name() + " doesn't have a pin type.";
      }
    }

    LOG_INFO << "Pins Num : " << num_pins;
    LOG_INFO << "1. IO Ports Num : " << num_io_ports;
    LOG_INFO << "2. Instance Ports Num : " << num_instance_ports;
    LOG_INFO << "3. Fake Pins Num : " << num_fake_pins;
  }

  void PlacerDB::printRegionInfo() const
  {
    Design* design = this->get_design();

    int32_t num_regions = 0;
    num_regions = design->get_region_list().size();

    LOG_INFO << "Regions Num : " << num_regions;
  }

  void PlacerDB::saveVerilogForDebug(std::string path)
  {
    _db_wrapper->saveVerilogForDebug(path);
  }

  // private
  PlacerDB* PlacerDB::_s_placer_db_instance = nullptr;

}  // namespace ipl