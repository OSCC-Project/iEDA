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
 * @File Name: dm_design_net.cpp
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
#include "ista_io.h"
#include "tool_manager.h"

namespace idm {
/**
 * @Brief : calculate total wire length for all net list
 * @return int64_t
 */
uint64_t DataManager::maxFanout()
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr == nullptr) {
    return 0;
  }

  return net_list_ptr == nullptr ? 0 : net_list_ptr->maxFanout();
}

uint64_t DataManager::allNetLength()
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr == nullptr) {
    return 0;
  }

  return netListLength(net_list_ptr->get_net_list());
}

/**
 * @Brief : calculate wire length for net
 * @param  net_name
 * @return int64_t
 */
uint64_t DataManager::netLength(string net_name)
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr == nullptr) {
    return 0;
  }

  IdbNet* net = net_list_ptr->find_net(net_name);
  return net == nullptr ? 0 : net->wireLength();
}

/**
 * @Brief : calculate the total wire length for net list
 * @param  net_list
 * @return int64_t
 */
uint64_t DataManager::netListLength(vector<IdbNet*>& net_list)
{
  uint64_t net_len = 0;
  for (auto net : net_list) {
    net_len += net->wireLength();
  }

  return net_len;
}

/**
 * @Brief : calculate the total wire length for net list
 * @param  net_list
 * @return int64_t
 */
uint64_t DataManager::netListLength(vector<string>& net_name_list)
{
  uint64_t net_len = 0;
  for (auto net_name : net_name_list) {
    net_len += netLength(net_name);
  }

  return net_len;
}

/**
 * @Brief : set IO pin to net
 * @param  io_pin_name
 * @param  net_name
 * @return true
 * @return false
 */
bool DataManager::setNetIO(string io_pin_name, string net_name)
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  IdbPins* pin_list_ptr = _design->get_io_pin_list();
  if (net_list_ptr == nullptr || pin_list_ptr == nullptr) {
    return 0;
  }

  IdbPin* io_pin = pin_list_ptr->find_pin(io_pin_name);
  IdbNet* net = net_list_ptr->find_net(net_name);
  if (io_pin == nullptr || net == nullptr) {
    return false;
  }

  net->add_io_pin(io_pin);

  return true;
}
/**
 * @Brief : get clock net list
 * @return vector<IdbNet*>
 */
vector<IdbNet*> DataManager::getClockNetList()
{
  vector<IdbNet*> net_list;

  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr != nullptr) {
    for (auto net : net_list_ptr->get_net_list()) {
      if (net->is_clock()) {
        net_list.emplace_back(net);
      }
    }
  }

  return net_list;
}

/**
 * @Brief : get signal net list
 * @return vector<IdbNet*>
 */
vector<IdbNet*> DataManager::getSignalNetList()
{
  vector<IdbNet*> net_list;

  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr != nullptr) {
    for (auto net : net_list_ptr->get_net_list()) {
      if (net->is_signal() || net->get_connect_type() == IdbConnectType::kNone) {
        net_list.emplace_back(net);
      }
    }
  }

  return net_list;
}

/**
 * @Brief : get pdn net list
 * @return vector<IdbNet*>
 */
vector<IdbNet*> DataManager::getPdnNetList()
{
  vector<IdbNet*> net_list;

  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr != nullptr) {
    for (auto net : net_list_ptr->get_net_list()) {
      if (net->is_pdn()) {
        net_list.emplace_back(net);
      }
    }
  }

  return net_list;
}

/**
 * @Brief : get net list that contains IO Pins
 * @return vector<IdbNet*>
 */
vector<IdbNet*> DataManager::getIONetList()
{
  vector<IdbNet*> net_list;

  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr != nullptr) {
    for (auto net : net_list_ptr->get_net_list()) {
      /// IO Pin Exist
      if (net->has_io_pins()) {
        net_list.emplace_back(net);
      }
    }
  }

  return net_list;
}

IdbPin* DataManager::getDriverOfNet(IdbNet* net)
{
  return net->get_driving_pin();
}

uint64_t DataManager::getClockNetListLength()
{
  auto net_list = getClockNetList();
  return netListLength(net_list);
}
uint64_t DataManager::getSignalNetListLength()
{
  auto net_list = getSignalNetList();
  return netListLength(net_list);
}
uint64_t DataManager::getPdnNetListLength()
{
  auto net_list = getPdnNetList();
  return netListLength(net_list);
}
uint64_t DataManager::getIONetListLength()
{
  auto net_list = getIONetList();
  return netListLength(net_list);
}

IdbNet* DataManager::createNet(const string& net_name, IdbConnectType type)
{
  auto* netlist = _design->get_net_list();

  auto* net = netlist->add_net(net_name, type);
  return net;
}

bool DataManager::disconnectNet(IdbNet* net)
{
  return true;
}

bool DataManager::connectNet(IdbNet* net)
{
  return true;
}

bool DataManager::setNetType(string net_name, string type)
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  auto net = net_list_ptr->find_net(net_name);
  if (net != nullptr) {
    net->set_connect_type(type);
    return true;
  }

  return false;
}

IdbInstance* DataManager::getIoCellByIoPin(IdbPin* io_pin)
{
  IdbNet* net = io_pin->get_net();
  if (net == nullptr) {
    std::cout << "Error : can not find net for IO pin " << io_pin->get_pin_name() << std::endl;
    return nullptr;
  }

  /// if the net connect io pin to instance pin, there are only 2 pins in 1 net
  for (IdbPin* pin : net->get_instance_pin_list()->get_pin_list()) {
    /// find the instance pin
    if (pin->get_pin_name() != io_pin->get_pin_name()) {
      return pin->get_instance();
    }
  }

  return nullptr;
}
/**
 * @brief get all the clock net name list for this design
 *
 * @return vector<string>
 */
vector<string> DataManager::getClockNetNameList()
{
  vector<string> clock_name_List;

  return staInst->getClockNetNameList();
}
/**
 * @brief check if net is a clock net
 *
 * @param net_name
 * @return true
 * @return false
 */
bool DataManager::isClockNet(string net_name)
{
  return staInst->isClockNet(net_name);
}

}  // namespace idm
