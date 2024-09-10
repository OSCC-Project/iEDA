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
 * @file		IdbLayer.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Tech Layer information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbLayer.h"

#include <algorithm>
#include <cctype>

#include "IdbTrackGrid.h"

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Layer type.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Layer.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbLayer::IdbLayer()
{
  _name = "";
  _type = IdbLayerType::kNone;
  _layer_id = 0;
  _layer_order = 0;
}

void IdbLayer::set_type(string type)
{
  _type = IdbEnum::GetInstance()->get_layer_property()->get_type(type);
}

void IdbLayer::print()
{
  std::cout << "name =  " << _name << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @date		25/05/2021
 * @version		0.1
* @description


        Construct Layers.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbLayers::IdbLayers()
{
  _routing_layer_index = 0;
  _cut_layer_index = 0;
  _z_order = 0;
}

IdbLayers::~IdbLayers()
{
  reset_layers();
}

void IdbLayers::reset_layers()
{
  for (auto& layer : _layers) {
    if (layer != nullptr) {
      delete layer;
      layer = nullptr;
    }
  }

  _layers.clear();
}

IdbLayer* IdbLayers::set_layer(string layer_name, string type)
{
  IdbLayer* layer_find = find_layer(layer_name, true);
  if (nullptr == layer_find) {
    IdbLayerType new_type = IdbEnum::GetInstance()->get_layer_property()->get_type(type);

    switch (new_type) {
      case IdbLayerType::kLayerCut: {
        layer_find = dynamic_cast<IdbLayer*>(new IdbLayerCut());
        layer_find->set_id(_cut_layer_index++);
        break;
      }
      case IdbLayerType::kLayerImplant: {
        layer_find = dynamic_cast<IdbLayer*>(new IdbLayerImplant());
        break;
      }
      case IdbLayerType::kLayerMasterslice: {
        layer_find = dynamic_cast<IdbLayer*>(new IdbLayerMasterslice());
        break;
      }
      case IdbLayerType::kLayerOverlap: {
        layer_find = dynamic_cast<IdbLayer*>(new IdbLayerOverlap());
        break;
      }
      case IdbLayerType::kLayerRouting: {
        layer_find = dynamic_cast<IdbLayer*>(new IdbLayerRouting());
        layer_find->set_id(_routing_layer_index++);
        break;
      }

      default:
        layer_find = nullptr;
        break;
    }

    if (layer_find) {
      layer_find->set_name(layer_name);
      layer_find->set_order(_z_order);
      _layers.emplace_back(layer_find);
    }

    _z_order++;
  }

  // std::cout << "Routing layer id = " << _routing_layer_index << std::endl;

  return layer_find;
}  // namespace idb

IdbLayer* IdbLayers::find_layer(const string& src_name, bool new_layer)
{
  auto equalIgnoreCase = [](const std::string& str1, const std::string& str2) {
    if (str1.size() != str2.size()) {
      return false;
    }
    for (size_t i = 0; i < str1.size(); ++i) {
      if (toupper(str1[i]) != toupper(str2[i])) {
        return false;
      }
    }
    return true;
  };
  for (IdbLayer* layer : _layers) {
    if (equalIgnoreCase(src_name, layer->get_name())) {
      return layer;
    }
  }

  //   if (!new_layer) {
  //     std::cout << "[IdbLayer Error] : can not find layer = " << src_name << std::endl;
  //   }

  return nullptr;
}

IdbLayer* IdbLayers::find_layer(IdbLayer* src_layer)
{
  string name = src_layer->get_name();
  return find_layer(name);
}

IdbLayer* IdbLayers::find_routing_layer(uint32_t index)
{
  if (_routing_layers.size() > index) {
    return _routing_layers.at(index);
  }

  return nullptr;
}

int32_t IdbLayers::get_layer_order(string layer_name)
{
  for (int32_t i = 0; i < get_layers_num(); ++i) {
    if (_layers[i] != nullptr && _layers[i]->compareLayer(layer_name)) {
      return i;
    }
  }

  return -1;
}

int32_t IdbLayers::get_layer_order(IdbLayer* layer)
{
  if (layer == nullptr) {
    return -1;
  }

  for (int32_t i = 0; i < get_layers_num(); ++i) {
    if (_layers[i] != nullptr && _layers[i]->compareLayer(layer->get_name())) {
      return i;
    }
  }

  return -1;
}

void IdbLayers::print()
{
  for (IdbLayer* layer : _layers) {
    layer->print();
  }
}

/// find the layer between layer 1 and layer 2
IdbLayer* IdbLayers::find_middle_layer(string layer_name_1, string layer_name_2)
{
  IdbLayer* layer_1 = find_layer(layer_name_1);
  IdbLayer* layer_2 = find_layer(layer_name_2);
  if (layer_1 == nullptr || layer_2 == nullptr) {
    return nullptr;
  }

  /// get the order of middle layer
  uint8_t find_layer_order = (layer_1->get_order() + layer_2->get_order()) / 2;
  return find_layer_by_order(find_layer_order);
}

IdbLayer* IdbLayers::find_layer_by_order(uint8_t order)
{
  for (IdbLayer* layer : _layers) {
    if (layer->get_order() == order) {
      return layer;
    }
  }

  std::cout << "[IdbLayer Error] : can not find layer with order = " << order << std::endl;

  return nullptr;
}

vector<IdbLayerCut*> IdbLayers::find_cut_layer_list(string layer_name_1, string layer_name_2)
{
  vector<IdbLayerCut*> cut_layer_list;

  IdbLayer* layer_1 = find_layer(layer_name_1);
  IdbLayer* layer_2 = find_layer(layer_name_2);
  if (layer_1 == nullptr || layer_2 == nullptr) {
    return cut_layer_list;
  }

  /// get the order of middle layer
  int32_t order_min = std::min(layer_1->get_order(), layer_2->get_order());
  int32_t order_max = std::max(layer_1->get_order(), layer_2->get_order());
  for (int i = order_min + 1; i < order_max; i++) {
    IdbLayer* layer_find = find_layer_by_order(i);
    if (layer_find->is_cut()) {
      cut_layer_list.emplace_back(dynamic_cast<IdbLayerCut*>(layer_find));
    }
  }

  return cut_layer_list;
}

/**
 * @brief return a minspacing with given width and parallel_length
 a example from lefdef5.8
 *
 SPACINGTABLE
PARALLELRUNLENGTH 0.00 0.50 3.00 5.00
#lengths must be increasing
WIDTH 0.000.15 0.15 0.15 0.15#max width>0.00
WIDTH 0.250.15 0.20 0.20 0.20#max width>0.25
WIDTH 1.500.15 0.50 0.50 0.50#max width>1.50
WIDTH 3.000.15 0.50 1.00 1.00#max width>3.00        ==> width 5.000 stands here
WIDTH 5.000.15 0.50 1.00 2.00 ;#max width>5.00

 * @param width
 * @param parallel_length
 * @return int32_t
 */
int32_t IdbParallelSpacingTable::get_spacing(int32_t width, int32_t parallel_length)
{
  static auto search = [](vector<int32_t>& arr, int32_t target) {
    ssize_t l = 0;
    ssize_t r = arr.size() - 1;
    while (l <= r) {
      ssize_t m = (l + r) / 2;
      if (arr[m] == target) {
        return m - 1 >= 0 ? m - 1 : 0;
      }
      if (arr[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }
    return r;
  };

  ssize_t iwidth = search(_width, width);
  ssize_t ilength = search(_parallel_run_length, parallel_length);
  return _spacing.at(iwidth).at(ilength);
}

}  // namespace idb
