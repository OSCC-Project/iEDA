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
/**
 * @File Name: dm_design_inst.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {

/**
 * @Brief :
 * @return int64_t
 */
double DataManager::instanceArea(IdbInstanceType type)
{
  uint64_t inst_area = 0;
  IdbInstanceList* inst_list = _design->get_instance_list();
  if (inst_list == nullptr) {
    return inst_area;
  }

  for (auto inst : inst_list->get_instance_list()) {
    if (type == IdbInstanceType::kMax || inst->get_type() == type) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
      continue;
    }

    if (type == IdbInstanceType::kNetlist && IdbInstanceType::kNone == inst->get_type()) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
      continue;
    }
  }

  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();
  return ((double) inst_area) / (dbu * dbu);
}

double DataManager::distInstArea()
{
  return instanceArea(IdbInstanceType::kDist);
}

double DataManager::netlistInstArea()
{
  return instanceArea(IdbInstanceType::kNetlist);
}

double DataManager::timingInstArea()
{
  return instanceArea(IdbInstanceType::kTiming);
}
/**
 * @Brief : get pin number for instance
 * @param  inst_name
 * @return int32_t
 */
int32_t DataManager::instancePinNum(string inst_name)
{
  IdbInstanceList* inst_list = _design->get_instance_list();
  IdbInstance* inst = inst_list->find_instance(inst_name);
  if (inst != nullptr) {
    IdbPins* pin_list = inst->get_pin_list();
    if (pin_list != nullptr) {
      return pin_list->get_pin_num();
    }
  }

  return -1;
}

/**
 * @Brief : create instance
 * @param  inst_name instance name created
 * @param  cell_master_name name of cell master to describe property
 * @param  coord_x coordinate of x
 * @param  coord_y coordinate of y
 * @param  orient Specify orientation
 * @param  type Specifies the source of the instance
 * @param  status Specifies the instance placement status
 * @return IdbInstance*
 */
IdbInstance* DataManager::createInstance(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                                         IdbInstanceType type, IdbPlacementStatus status)
{
  IdbInstanceList* inst_list = _design->get_instance_list();
  IdbCellMasterList* cell_master_list = _layout->get_cell_master_list();

  IdbInstance* inst = inst_list->add_instance(inst_name);
  if (inst != nullptr) {
    IdbCellMaster* cell_master = cell_master_list->find_cell_master(cell_master_name);
    if (cell_master == nullptr) {
      return nullptr;
    }

    inst->set_cell_master(cell_master);
    inst->set_type(type);
    /// do not update other cell data, and set update false.
    inst->set_orient(orient, false);
    inst->set_coodinate(coord_x, coord_y);
    inst->set_status(status);
  }
  return inst;
}

/**
 * @Brief : Instance is specified in the original netlist.
            This is the default value, and is normally NOT written out in the DEF file.
 * @param  inst_name
 * @param  cell_master_name
 * @param  coord_x
 * @param  coord_y
 * @param  orient
 * @param  status
 * @return IdbInstance*
 */
IdbInstance* DataManager::createNetlistInst(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                                            IdbPlacementStatus status)
{
  return createInstance(inst_name, cell_master_name, coord_x, coord_y, orient, IdbInstanceType::kNetlist, status);
}

/**
 * @Brief : create instance that is a physical cell (that is, it only connects to power or ground nets),
            such as filler cells, well-taps, and decoupling caps.
 * @param  inst_name
 * @param  cell_master_name
 * @param  coord_x
 * @param  coord_y
 * @param  orient
 * @param  status
 * @return IdbInstance*
 */
IdbInstance* DataManager::createPhysicalInst(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                                             IdbPlacementStatus status)
{
  return createInstance(inst_name, cell_master_name, coord_x, coord_y, orient, IdbInstanceType::kDist, status);
}
/**
 * @Brief : Instance is a logical rather than physical change to the netlist, and is typically used as
            a BUFFER for a clock-tree, or to improve timing on long nets.
 * @param  inst_name
 * @param  cell_master_name
 * @param  coord_x
 * @param  coord_y
 * @param  orient
 * @param  status
 * @return IdbInstance*
 */
IdbInstance* DataManager::createTimingInst(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                                           IdbPlacementStatus status)
{
  return createInstance(inst_name, cell_master_name, coord_x, coord_y, orient, IdbInstanceType::kTiming, status);
}
/**
 * @Brief : Instance is generated by the user for some user-defined reason.
 * @param  inst_name
 * @param  cell_master_name
 * @param  coord_x
 * @param  coord_y
 * @param  orient
 * @param  status
 * @return IdbInstance*
 */
IdbInstance* DataManager::createUserInst(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient,
                                         IdbPlacementStatus status)
{
  return createInstance(inst_name, cell_master_name, coord_x, coord_y, orient, IdbInstanceType::kUser, status);
}
/**
 * @Brief : insert buffer to net
 * @param  inst_name
 * @param  cell_master_name
 * @param  net_name
 * @param  pin_name_list
 * @return true
 * @return false
 */
IdbInstance* DataManager::insertBufferToNet(string inst_name, string cell_master_name, string net_name, vector<string> pin_name_list)
{
  /// check net
  IdbNet* net = _design->get_net_list()->find_net(net_name);
  if (net == nullptr) {
    return nullptr;
  }

  /// check pin
  IdbCellMasterList* cell_master_ptr = _layout->get_cell_master_list();
  IdbCellMaster* cell_master = cell_master_ptr->find_cell_master(cell_master_name);
  if (cell_master == nullptr) {
    return nullptr;
  }
  vector<IdbTerm*> term_list = cell_master->get_term_list();
  for (auto pin_name : pin_name_list) {
    bool b_find = false;
    for (auto term : term_list) {
      if (term->get_name().compare(pin_name) == 0) {
        b_find = true;
        break;
      }
    }
    /// this pin can not found in master
    if (!b_find) {
      std::cout << "[IDM Error] can not find pin " << pin_name << " in master " << cell_master_name << std::endl;
      return nullptr;
    }
  }

  /// check instance
  IdbInstance* inst = createTimingInst(inst_name, cell_master_name);
  if (inst == nullptr) {
    return nullptr;
  }

  ///  add pin to net
  for (auto pin_name : pin_name_list) {
    IdbPin* pin = inst->get_pin_by_term(pin_name);
    if (pin == nullptr) {
      break;
    }

    net->add_instance_pin(pin);
  }

  return inst;
}

/**
 * @Brief : add filler for core in coordinate(x, y)
 * @param  inst_name
 * @param  cell_master_name
 * @param  coord_x
 * @param  coord_y
 * @return IdbInstance*
 */
IdbInstance* DataManager::insertCoreFiller(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y)
{
  IdbOrient orient = getDefaultOrient(coord_x, coord_y);
  return createPhysicalInst(inst_name, cell_master_name, coord_x, coord_y, orient, IdbPlacementStatus::kPlaced);
}

IdbInstance* DataManager::insertIOFiller(string inst_name, string cell_master_name, int32_t coord_x, int32_t coord_y, IdbOrient orient)
{
  // IdbOrient orient = getDefaultOrient(coord_x, coord_y);
  return createPhysicalInst(inst_name, cell_master_name, coord_x, coord_y, orient, IdbPlacementStatus::kPlaced);
}

/**
 * @brief
 *
 * @param inst_name
 * @param x
 * @param y
 * @param orient_name
 * @param cell_master_name
 * @param source
 * @return true
 * @return false
 */
bool DataManager::placeInst(string inst_name, int32_t x, int32_t y, string orient_name, string cell_master_name, string source)
{
  IdbCellMaster* cellmaster;
  if (cell_master_name == "") {
    auto inst = _design->get_instance_list()->find_instance(inst_name);
    cellmaster = inst->get_cell_master();
  } else {
    cellmaster = _layout->get_cell_master_list()->find_cell_master(cell_master_name);
  }

  IdbOrient orient = IdbEnum::GetInstance()->get_site_property()->get_orient_value(orient_name);
  if (cellmaster == nullptr || orient == IdbOrient::kNone) {
    std::cout << "[IDM Error] inst_name = " << inst_name << " cell_master_name = " << cell_master_name << " orient_name = " << orient_name
              << std::endl;
    return false;
  }

  int32_t width = cellmaster->get_width();
  int32_t height = cellmaster->get_height();
  int32_t urx;
  int32_t ury;
  if (orient == IdbOrient::kN_R0 || orient == IdbOrient::kS_R180 || orient == IdbOrient::kFN_MY || orient == IdbOrient::kFS_MX) {
    urx = x + width;
    ury = y + height;
  } else {
    urx = x + height;
    ury = y + height;
  }

  if (cellmaster->is_endcap()) {
    if (!isOnDieBoundary(x, y, urx, ury, orient)) {
      printf("%s info have problem\n", inst_name.c_str());
    }
  } else if (cellmaster->is_pad() || cellmaster->is_pad_filler()) {
    bool can_place = checkInstPlacer(x, y, urx, ury, orient);
    if (!can_place) {
      printf("%s info have problem\n", inst_name.c_str());
    }
  }

  auto idb_inst_list = _design->get_instance_list();
  IdbInstance* instance = idb_inst_list->find_instance(inst_name);
  if (instance == nullptr) {
    instance = new IdbInstance();
    instance->set_name(inst_name);
    instance->set_cell_master(cellmaster);
    idb_inst_list->add_instance(instance);
  }

  if (!source.empty()) {
    instance->set_type(source);
  }

  instance->set_orient(orient);
  instance->set_coodinate(x, y);
  instance->set_status_fixed();

  return true;
}

}  // namespace idm
