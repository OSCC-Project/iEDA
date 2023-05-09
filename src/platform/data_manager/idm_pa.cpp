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
 * @File Name: dm_pa.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "base/FileHeader.h"
#include "flow_config.h"
#include "idm.h"

namespace idm {

/**
 * @brief build pa data
 *
 * @param pa_config_path
 * @return true
 * @return false
 */
bool DataManager::buildPA(const std::map<std::string, std::map<std::string, std::vector<ids::AccessPoint>>>& master_access_point_map)
{
  _master_access_point_map = master_access_point_map;
  wrapPA();

  return true;
}

/**
 * @brief wrap first pa in pa list (pa[0]) as pin coordinate for each pin of net
 *
 * @return true
 * @return false
 */
bool DataManager::wrapPA()
{
  auto net_list_ptr = _design->get_net_list();
  if (net_list_ptr == nullptr) {
    return false;
  }

  for (auto net : net_list_ptr->get_net_list()) {
    if (nullptr == net || nullptr == net->get_instance_pin_list())
      continue;

    for (IdbPin* idb_pin : net->get_instance_pin_list()->get_pin_list()) {
      auto pa_list = getInstancePaPointList(idb_pin->get_instance()->get_name(), idb_pin->get_pin_name());

      if (pa_list.size() > 0) {
        /// if access point exist, save the first access point to IdbPin
        auto access_point = pa_list[0];
        idb_pin->set_grid_coordinate(access_point.x, access_point.y);
      } else {
        std::cout << "[idm error] PA do not exist for net = " << net->get_net_name() << " pin =  " << idb_pin->get_pin_name() << std::endl;
      }
    }
  }

  return true;
}

std::vector<ids::AccessPoint> DataManager::getMasterPaPointList(std::string master_name, std::string pin_name)
{
  auto master_iter = _master_access_point_map.find(master_name);
  if (master_iter == _master_access_point_map.end()) {
    std::cout << "[idm Wraning] Can not find cell master = " << master_name << std::endl;
  }
  auto pin_iter = (master_iter->second).find(pin_name);
  if (pin_iter == (master_iter->second).end()) {
    std::cout << "[idm Wraning] Can not find pin = " << pin_name << " in cell master = " << master_name << std::endl;
    return {};
  }

  return pin_iter->second;
}
/**
 * @brief get pa list by instance name & pin name
 *
 * @param instance_name
 * @param pin_name
 * @return std::vector<ids::AccessPoint>
 */
std::vector<ids::AccessPoint> DataManager::getInstancePaPointList(std::string instance_name, std::string pin_name)
{
  std::vector<ids::AccessPoint> access_point_list;

  auto idb_inst_list = _design->get_instance_list();
  auto idb_inst = idb_inst_list->find_instance(instance_name);
  if (idb_inst != nullptr) {
    /// get pa list in cell master
    access_point_list = getMasterPaPointList(idb_inst->get_cell_master()->get_name(), pin_name);
    /// transform pa list in cell master by instance coordinate & orient
    for (auto& pa : access_point_list) {
      idb_inst->transformCoordinate(pa.x, pa.y);

      auto inst_boundary = idb_inst->get_bounding_box();
      if (pa.x > inst_boundary->get_high_x() || pa.x < inst_boundary->get_low_x() || pa.y > inst_boundary->get_high_y()
          || pa.y < inst_boundary->get_low_y()) {
        std::cout << "[idm error] PA x = " << pa.x << " y = " << pa.y << " is not inside the instance " << idb_inst->get_name() << " ( "
                  << inst_boundary->get_low_x() << ", " << inst_boundary->get_low_y() << ") ( " << inst_boundary->get_high_x() << ", "
                  << inst_boundary->get_high_y() << ") " << std::endl;
      }
    }
  }

  return access_point_list;
}

/**
 * @brief get pa list by cell master name & pin name with target position & target orient
 *
 * @param cell_master_name cell master name
 * @param pin_name pin name
 * @param inst_x target coordinate x
 * @param inst_y target coordinate y
 * @param idb_orient target coordinate orient
 * @return std::vector<ids::AccessPoint>
 */
std::vector<ids::AccessPoint> DataManager::getInstancePaPointList(std::string cell_master_name, std::string pin_name, int32_t inst_x,
                                                                  int32_t inst_y, idb::IdbOrient idb_orient)
{
  /// get pa list in cell master
  std::vector<ids::AccessPoint> access_point_list = getMasterPaPointList(cell_master_name, pin_name);
  /// transform pa list in cell master by coordinate & orient
  for (auto& pa : access_point_list) {
    transformCoordinate(pa.x, pa.y, cell_master_name, inst_x, inst_y, idb_orient);
  }
  return access_point_list;
}

}  // namespace idm
