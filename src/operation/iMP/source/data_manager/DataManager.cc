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
 * @FilePath: /irefactor/src/operation/iPL/source/DataManager.cc
 * Contact : https://github.com/sjchanson
 */

#include "DataManager.hh"

#include <iostream>

#include "IDBWrapper.hh"
#include "Rectangle.hh"

namespace imp {

DataManager::DataManager() : _db_wrapper(nullptr)
{
}

void DataManager::readFormLefDef(const std::string& json_path)
{
  setDbWrapper(new IDBWrapper(json_path));
}

DataManager::~DataManager()
{
  if (_db_wrapper) {
    delete _db_wrapper;
  }
}

void DataManager::setDbWrapper(DBWrapper* db_wrapper)
{
  _db_wrapper = db_wrapper;
  printDataManager();
}

const Layout* DataManager::get_layout() const
{
  return _db_wrapper->get_layout();
}

Design* DataManager::get_design() const
{
  return _db_wrapper->get_design();
}

void DataManager::updateFromSourceDataBase()
{
  _db_wrapper->updateFromSourceDataBase();
}

// void DataManager::updateFromSourceDataBase(std::vector<std::string> inst_list)
// {
//   _db_wrapper->updateFromSourceDataBase(inst_list);
// }

// void DataManager::updateInstancesForDebug(std::vector<Instance*> inst_list)
// {
//   auto* design = this->get_design();
//   for (auto* inst : inst_list) {
//     design->add_instance(inst);
//   }
// }

void DataManager::printDataManager() const
{
  printLayoutInfo();
  printInstanceInfo();
  printNetInfo();
  printPinInfo();
  printRegionInfo();
}

void DataManager::printLayoutInfo() const
{
  Design* design = this->get_design();
  const Layout* layout = this->get_layout();

  std::string design_name = design->get_design_name();
  int32_t database_unit = layout->get_database_unit();
  Rectangle<int32_t> die_rect = layout->get_die_shape();
  Rectangle<int32_t> core_rect = layout->get_core_shape();
  int32_t row_height = layout->get_row_height();
  int32_t site_width = layout->get_site_width();

  std::cout << "Design name : " << design_name;
  std::cout << "Database unit : " << database_unit;
  std::cout << "Die rectangle : " << die_rect.get_ll_x() << "," << die_rect.get_ll_y() << " " << die_rect.get_ur_x() << ","
            << die_rect.get_ur_y();
  std::cout << "Core rectangle : " << core_rect.get_ll_x() << "," << core_rect.get_ll_y() << " " << core_rect.get_ur_x() << ","
            << core_rect.get_ur_y();
  std::cout << "Row height : " << row_height;
  std::cout << "Site width : " << site_width;

  int64_t core_area = static_cast<int64_t>(core_rect.get_width()) * static_cast<int64_t>(core_rect.get_height());
  int64_t place_instance_area = 0;
  int64_t non_place_instance_area = 0;

  for (auto* inst : design->get_instance_list()) {
    // Ignore the insts outside the core.
    if (inst->isOutsideInstance() && inst->isFixed()) {
      continue;
    }

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
      non_place_instance_area += inst_width * inst_height;
    } else {
      place_instance_area += inst_width * inst_height;
    }
  }

  // TODO : exclude the overlap region.
  for (auto* blockage : design->get_region_list()) {
    for (auto boundary : blockage->get_boundaries()) {
      int64_t boundary_width = static_cast<int64_t>(boundary.get_width());
      int64_t boundary_height = static_cast<int64_t>(boundary.get_height());
      non_place_instance_area += boundary_width * boundary_height;
    }
  }

  std::cout << "Core area : " << core_area;
  std::cout << "Non place instance area : " << non_place_instance_area;
  std::cout << "Place instance area : " << place_instance_area;

  double util = static_cast<double>(place_instance_area) / (core_area - non_place_instance_area) * 100;
  std::cout << "Uitization(%) : " << util;
  if (util > 100.1)
    std::cerr << "Utilization exceeds 100%";
}

float DataManager::obtainUtilization()
{
  Design* design = this->get_design();
  const Layout* layout = this->get_layout();
  Rectangle<int32_t> core_rect = layout->get_core_shape();
  int64_t core_area = static_cast<int64_t>(core_rect.get_width()) * static_cast<int64_t>(core_rect.get_height());
  int64_t place_instance_area = 0;
  int64_t non_place_instance_area = 0;

  for (auto* inst : design->get_instance_list()) {
    // Ignore the insts outside the core.
    if (inst->isOutsideInstance() && inst->isFixed()) {
      continue;
    }
    int64_t inst_width = static_cast<int64_t>(inst->get_shape().get_width());
    int64_t inst_height = static_cast<int64_t>(inst->get_shape().get_height());
    if (inst->isFixed()) {
      non_place_instance_area += inst_width * inst_height;
    } else {
      place_instance_area += inst_width * inst_height;
    }
  }

  // TODO : exclude the overlap region.
  for (auto* blockage : design->get_region_list()) {
    for (auto boundary : blockage->get_boundaries()) {
      int64_t boundary_width = static_cast<int64_t>(boundary.get_width());
      int64_t boundary_height = static_cast<int64_t>(boundary.get_height());
      non_place_instance_area += boundary_width * boundary_height;
    }
  }

  float util = static_cast<float>(place_instance_area) / (core_area - non_place_instance_area);
  return util;
}

void DataManager::printInstanceInfo() const
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
      } else if (cell_master->isLogic()) {
        num_logic_insts++;
      } else if (cell_master->isFlipflop()) {
        num_flipflop_cells++;
      } else if (cell_master->isClockBuffer()) {
        num_clock_buffers++;
      } else if (cell_master->isLogicBuffer()) {
        num_logic_buffers++;
      } else if (cell_master->isIOCell()) {
        num_io_cells++;
      } else if (cell_master->isPhysicalFiller()) {
        num_physical_insts++;
      } else {
        std::cerr << "Instance : " + inst->get_name() + " doesn't have a cell type.";
      }
    }

    if (inst->isFakeInstance()) {
      num_fake_instances++;
    } else if (inst->isNormalInstance()) {
      //
    } else if (inst->isOutsideInstance()) {
      num_outside_insts++;
    } else {
      // std::cerr << "Instance : " + inst->get_name() + " doesn't have a instance type.";
    }

    if (inst->isUnPlaced()) {
      num_unplaced_instances++;
    } else if (inst->isPlaced()) {
      num_placed_instances++;
    } else if (inst->isFixed()) {
      num_fixed_instances++;
    } else {
      std::cerr << "Instance : " + inst->get_name() + " doesn't have a instance state.";
    }
  }

  std::cout << "Instances Num : " << num_instances;
  std::cout << "1. Macro Num : " << num_macros;
  std::cout << "2. Stdcell Num : " << num_instances - num_macros;
  std::cout << "2.1 Logic Instances : " << num_logic_insts;
  std::cout << "2.2 Flipflops : " << num_flipflop_cells;
  std::cout << "2.3 Clock Buffers : " << num_clock_buffers;
  std::cout << "2.4 Logic Buffers : " << num_logic_buffers;
  std::cout << "2.5 IO Cells : " << num_io_cells;
  std::cout << "2.6 Physical Instances : " << num_physical_insts;
  std::cout << "Core Outside Instances : " << num_outside_insts;
  std::cout << "Fake Instances : " << num_fake_instances;
  std::cout << "Unplaced Instances Num : " << num_unplaced_instances;
  std::cout << "Placed Instances Num : " << num_placed_instances;
  std::cout << "Fixed Instances Num : " << num_fixed_instances;
  std::cout << "Optional CellMaster Num : " << num_cell_masters;
}

void DataManager::printNetInfo() const
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
    } else if (net->isResetNet()) {
      num_reset_nets++;
    } else if (net->isSignalNet()) {
      num_signal_nets++;
    } else if (net->isFakeNet()) {
      num_fake_nets++;
    } else {
      num_no_type_nets++;
    }

    if (net->isNormalStateNet()) {
      num_normal_nets++;
    } else if (net->isDontCareNet()) {
      num_dontcare_nets++;
    } else {
      num_no_state_nets++;
    }
  }

  std::cout << "Nets Num : " << num_nets;
  std::cout << "1. ClockNets Num : " << num_clock_nets;
  std::cout << "2. ResetNets Num : " << num_reset_nets;
  std::cout << "3. SignalNets Num : " << num_signal_nets;
  std::cout << "4. FakeNets Num : " << num_fake_nets;
  std::cout << "Don't Care Net Num : " << num_dontcare_nets;

  if (num_no_type_nets != 0) {
    std::cerr << "Existed Nets don't have NET_TYPE : " << num_no_type_nets;
  }
  if (num_no_state_nets != 0) {
    std::cerr << "Existed Nets don't have NET_STATE : " << num_no_state_nets;
  }
}

void DataManager::printPinInfo() const
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
    } else if (pin->isInstancePort()) {
      num_instance_ports++;
    } else if (pin->isFakePin()) {
      num_fake_pins++;
    } else {
      std::cerr << "Pin : " + pin->get_name() + " doesn't have a pin type.";
    }
  }

  std::cout << "Pins Num : " << num_pins;
  std::cout << "1. IO Ports Num : " << num_io_ports;
  std::cout << "2. Instance Ports Num : " << num_instance_ports;
  std::cout << "3. Fake Pins Num : " << num_fake_pins;
}

void DataManager::printRegionInfo() const
{
  Design* design = this->get_design();

  int32_t num_regions = 0;
  num_regions = design->get_region_list().size();

  std::cout << "Regions Num : " << num_regions;
}

// void DataManager::saveVerilogForDebug(std::string path)
// {
//   _db_wrapper->saveVerilogForDebug(path);
// }

// private

}  // namespace imp