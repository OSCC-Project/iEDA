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
 * @project		iDB
 * @file		IdbDesign.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe def.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbDesign.h"

namespace idb {

IdbDesign::IdbDesign(IdbLayout* layout)
{
  //!----tbd----------
  _layout = layout;
  // new method will be replaced in future
  _design_name = "";
  _units = new IdbUnits();

  //  _row = new IdbRow();

  _instance_list = new IdbInstanceList();
  _io_pin_list = new IdbPins();
  _net_list = new IdbNetList();
  _via_list = new IdbVias();
  _blockage_list = new IdbBlockageList();
  _slot_list = new IdbSlotList();
  _group_list = new IdbGroupList();
  _special_net_list = new IdbSpecialNetList();
  _region_list = new IdbRegionList();
  _fill_list = new IdbFillList();
  _bus_bit_chars = new IdbBusBitChars();

  _bus_list = new IdbBusList();
}

IdbDesign::~IdbDesign()
{
  if (_instance_list != nullptr) {
    delete _instance_list;
    _instance_list = nullptr;
  }
  if (_io_pin_list != nullptr) {
    delete _io_pin_list;
    _io_pin_list = nullptr;
  }
  if (_net_list != nullptr) {
    delete _net_list;
    _net_list = nullptr;
  }
  if (_via_list != nullptr) {
    delete _via_list;
    _via_list = nullptr;
  }
  if (_blockage_list != nullptr) {
    delete _blockage_list;
    _blockage_list = nullptr;
  }
  if (_slot_list != nullptr) {
    delete _slot_list;
    _slot_list = nullptr;
  }
  if (_group_list != nullptr) {
    delete _group_list;
    _group_list = nullptr;
  }
  if (_special_net_list != nullptr) {
    delete _special_net_list;
    _special_net_list = nullptr;
  }
  if (_region_list != nullptr) {
    delete _region_list;
    _region_list = nullptr;
  }
  if (_fill_list != nullptr) {
    delete _fill_list;
    _fill_list = nullptr;
  }
  if (_bus_bit_chars != nullptr) {
    delete _bus_bit_chars;
    _bus_bit_chars = nullptr;
  }
  if (_bus_list != nullptr) {
    delete _bus_list;
    _bus_list = nullptr;
  }
}

// void IdbDesign::createDefaultVias(IdbLayers* layers)
// {
//   for (IdbLayer* layer : layers->get_cut_layers()) {
//     IdbLayerCut* cut_layer = dynamic_cast<IdbLayerCut*>(layer);
//     string via_name = cut_layer->get_name() + "_Generate";
//     _via_list->createViaDefault(via_name, cut_layer);
//   }
// }

bool IdbDesign::connectIOPinToPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer)
{
  if (point_list.size() < _POINT_MAX_ || layer == nullptr) {
    return false;
  }

  /// find the IO pin that covered by the point list
  IdbPin* pin = _io_pin_list->find_pin_by_coordinate_list(point_list, layer);
  if (pin == nullptr) {
    std::cout << "Error : no IO pin covered by point list." << std::endl;
    for (IdbCoordinate<int32_t>* pt : point_list) {
      std::cout << " ( " << pt->get_x() << " , " << pt->get_y() << " )";
    }
    std::cout << std::endl;
    return false;
  }

  /// if point list is not horizontal or vertical, gernerate the correct points and adjust the order
  if (point_list.size() == _POINT_MAX_ && point_list[0]->get_x() != point_list[1]->get_x()
      && point_list[0]->get_y() != point_list[1]->get_y()) {
    /// original value
    int32_t start_x = point_list[0]->get_x();
    int32_t start_y = point_list[0]->get_y();
    int32_t end_x = point_list[1]->get_x();
    int32_t end_y = point_list[1]->get_y();
    /// get the middle coordinate
    int32_t mid_x = (start_x + end_x) / 2;
    int32_t mid_y = (start_y + end_y) / 2;

    IdbCore* core = _layout->get_core();

    if (core->is_side_left_or_right(point_list[0]) || core->is_side_left_or_right(point_list[1])) {
      /// make horizontal
      IdbCoordinate<int32_t>* new_coordinate = new IdbCoordinate<int32_t>(mid_x, start_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new IdbCoordinate<int32_t>(mid_x, end_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);

    } else if (core->is_side_top_or_bottom(point_list[0]) || core->is_side_top_or_bottom(point_list[1])) {
      /// vertical
      IdbCoordinate<int32_t>* new_coordinate = new IdbCoordinate<int32_t>(start_x, mid_y);
      point_list.insert(point_list.begin() + 1, new_coordinate);
      new_coordinate = new IdbCoordinate<int32_t>(end_x, mid_y);
      point_list.insert(point_list.begin() + 2, new_coordinate);
    } else {
      std::cout << "Error : illegal point list." << std::endl;
      return false;
    }
  }

  return _special_net_list->connectIO(point_list, layer);
}

bool IdbDesign::connectPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, string net_name, string layer_name)
{
  return _special_net_list->addPowerStripe(point_list, net_name, layer_name);
}

}  // namespace idb
