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
/**
 * merge segment wire for all nets
 */
void DataManager::mergeNets()
{
  IdbNetList* net_list_ptr = _design->get_net_list();
  if (net_list_ptr != nullptr) {
#pragma omp parallel for schedule(dynamic)
    for (auto net : net_list_ptr->get_net_list()) {
      mergeNet(net);
    }
  }
}
/**
 * merge segment wire for net
 */
void DataManager::mergeNet(IdbNet* net)
{
  /// split segment into segment wire and segment via,
  /// keep_via : if true, delete points and use this segment as via
  /// return : nullpt or a new segment via
  auto split_segment = [](IdbRegularWireSegment* segment, bool keep_via) -> IdbRegularWireSegment* {
    if (keep_via) {
      /// delete wire points
      segment->clearPoints();
      return nullptr;
    } else {
      /// new a segment to save via
      IdbRegularWireSegment* via_seg = new IdbRegularWireSegment();
      via_seg->set_layer_status(true);
      via_seg->set_layer_name(segment->get_layer_name());
      via_seg->set_layer(segment->get_layer());
      via_seg->set_via_list(segment->get_via_list());
      via_seg->set_is_via(true);

      /// delete via in this segment
      segment->set_is_via(false);
      segment->set_via_list({});
      return via_seg;
    }
  };

  /// segment_map : segment map for each coordinate x or y
  /// b_horizontal : map is horizontal using coordiate y as key, otherwise using x as key
  /// std::vector<IdbRegularWireSegment*> : return segments that generated by merge, and need to add to net
  auto merge = [&](std::map<int, std::vector<IdbRegularWireSegment*>>& segment_map,
                   bool b_horizontal) -> std::pair<std::vector<IdbRegularWireSegment*>, std::vector<IdbRegularWireSegment*>> {
    std::vector<IdbRegularWireSegment*> new_segments;
    std::vector<IdbRegularWireSegment*> delete_segments;

    for (auto& [coordinate, segments] : segment_map) {
      if (segments.size() < 2) {
        /// no need to merge, ignore
        continue;
      }

      /// sort
      std::sort(segments.begin(), segments.end(), [&](IdbRegularWireSegment* a, IdbRegularWireSegment* b) {
        return b_horizontal ? a->get_point_start()->get_x() < b->get_point_start()->get_x()
                            : a->get_point_start()->get_y() < b->get_point_start()->get_y();
      });

      /// merge
      auto it_1 = segments.begin();
      auto it_2 = segments.begin() + 1;
      for (; it_2 != segments.end(); it_2++) {
        /// save point
        auto coord_1_begin = b_horizontal ? (*it_1)->get_point_start()->get_x() : (*it_1)->get_point_start()->get_y();
        auto coord_1_end = b_horizontal ? (*it_1)->get_point_second()->get_x() : (*it_1)->get_point_second()->get_y();
        auto coord_2_begin = b_horizontal ? (*it_2)->get_point_start()->get_x() : (*it_2)->get_point_start()->get_y();
        auto coord_2_end = b_horizontal ? (*it_2)->get_point_second()->get_x() : (*it_2)->get_point_second()->get_y();
        if (coord_1_end >= coord_2_begin) {
          /// interact or overlap, need to merge
          ///  step 1 : merge segment by extend it_1
          auto end_coord = std::max(coord_1_end, coord_2_end);
          b_horizontal ? (*it_1)->get_point_second()->set_x(end_coord) : (*it_1)->get_point_second()->set_y(end_coord);

          /// step 2 : if segment contains via, segment must be divided into two segment
          if ((*it_1)->is_via()) {
            /// new a segment as via segment
            IdbRegularWireSegment* via_seg = split_segment((*it_1), false);
            if (via_seg != nullptr) {
              new_segments.push_back(via_seg);
            }
          }

          /// step 3
          if ((*it_2)->is_via()) {
            /// keep segment and delete points
            split_segment((*it_2), true);
          } else {
            delete_segments.push_back((*it_2));
          }
        } else {
          /// no interact, move it_1 to it_2
          it_1 = it_2;
        }
      }
    }

    return std::make_pair(new_segments, delete_segments);
  };

  auto update_net = [&](std::vector<IdbRegularWireSegment*>& new_segments, std::vector<IdbRegularWireSegment*>& delete_segments) {
    /// delete useless segment
    for (auto* del_seg : delete_segments) {
      net->remove_segment(del_seg);
    }

    /// add new segment to wire 0
    if (new_segments.size() > 0) {
      auto& wire_segs = net->get_wire_list()->get_wire_list()[0]->get_segment_list();
      wire_segs.insert(wire_segs.end(), new_segments.begin(), new_segments.end());
    }
  };

  struct LayerData
  {
    std::map<int, std::vector<IdbRegularWireSegment*>> horizontal_map;
    std::map<int, std::vector<IdbRegularWireSegment*>> vertical_map;
  };
  std::map<int, LayerData> layer_map;

  for (auto* wire : net->get_wire_list()->get_wire_list()) {
    for (auto* segment : wire->get_segment_list()) {
      if (segment->is_wire()) {
        auto& layer_data = layer_map[segment->get_layer()->get_order()];

        /// classify segment to horizontal and vertical direction
        auto* point_start = segment->get_point_start();
        auto* point_end = segment->get_point_end();

        if (point_start->get_y() == point_end->get_y()) {
          /// horizontal
          auto& segment_list = layer_data.horizontal_map[point_start->get_y()];
          segment_list.emplace_back(segment);
        } else {
          /// vertical
          auto& segment_list = layer_data.vertical_map[point_start->get_x()];
          segment_list.emplace_back(segment);
        }
      }
    }
  }

  for (auto& [layer_id, layer_data] : layer_map) {
    auto [new_segs_h, del_segs_h] = merge(layer_data.horizontal_map, true);
    auto [new_segs_v, del_segs_v] = merge(layer_data.vertical_map, false);

    /// update net segments
    update_net(new_segs_h, del_segs_h);
    update_net(new_segs_v, del_segs_v);
  }
}

}  // namespace idm
