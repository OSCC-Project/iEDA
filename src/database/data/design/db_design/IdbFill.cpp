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
 * @file		IdbVias.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Vias information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbFill.h"

#include "IdbVias.h"

using namespace std;
namespace idb {

IdbFillLayer::IdbFillLayer()
{
  _layer = nullptr;
}

IdbFillLayer::~IdbFillLayer()
{
  reset_rect();
}

IdbRect* IdbFillLayer::get_rect(size_t index)
{
  if (index < _rect_list.size() && _rect_list.size() > 0) {
    return _rect_list.at(index);
  }

  return nullptr;
}

IdbRect* IdbFillLayer::add_rect()
{
  IdbRect* rect = new IdbRect();
  _rect_list.emplace_back(rect);

  return rect;
}

void IdbFillLayer::add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
  _rect_list.emplace_back(rect);
}

void IdbFillLayer::reset_rect()
{
  for (auto* rect : _rect_list) {
    if (rect != nullptr) {
      delete rect;
      rect = nullptr;
    }
  }

  _rect_list.clear();
  std::vector<IdbRect*>().swap(_rect_list);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbFillVia::IdbFillVia()
{
  _via = nullptr;
}

IdbFillVia::~IdbFillVia()
{
  if (_via != nullptr) {
    delete _via;
    _via = nullptr;
  }
}

IdbCoordinate<int32_t>* IdbFillVia::get_coordinate(size_t index)
{
  IdbCoordinate<int32_t>* coordinate = nullptr;
  if ((_coordinate_list.size() > index) && (index >= 0)) {
    coordinate = _coordinate_list.at(index);
  }

  return coordinate;
}

IdbCoordinate<int32_t>* IdbFillVia::add_coordinate(int32_t x, int32_t y)
{
  IdbCoordinate<int32_t>* coordinate = new IdbCoordinate<int32_t>(x, y);
  _coordinate_list.emplace_back(coordinate);

  return coordinate;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbFill::IdbFill()
{
  _layer = new IdbFillLayer();
  _via = new IdbFillVia();
}

IdbFill::~IdbFill()
{
  if (_layer != nullptr) {
    delete _layer;
    _layer = nullptr;
  }

  if (_via != nullptr) {
    delete _via;
    _via = nullptr;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbFillList::IdbFillList()
{
  _num_fill = 0;
}

IdbFillList::~IdbFillList()
{
  for (IdbFill* fill : _fill_list) {
    if (fill != nullptr) {
      delete fill;
      fill = nullptr;
    }
  }
  _fill_list.clear();
  std::vector<IdbFill*>().swap(_fill_list);
}

// IdbFill* IdbFillList::find_fill_by_layer(string name)
// {
//     for(IdbFill* fill : _fill_list)
//     {
//         if()
//         IdbLayer* layer = fill->get
//         if(fill->get_layer_name() == name)
//         {
//             return fill;
//         }
//     }

//     return nullptr;
// }

// IdbFill* IdbFillList::find_fill_by_via(string name)
// {
//     if(_num_vias > index)
//     {
//         return _via_list.at(index);
//     }

//     return nullptr;
// }
IdbFillLayer* IdbFillList::add_fill_layer(IdbLayer* layer)
{
  IdbFill* fill = new IdbFill();
  fill->set_type_layer();

  IdbFillLayer* fill_layer = fill->get_layer();
  fill_layer->set_layer(layer);

  _fill_list.emplace_back(fill);
  _num_fill++;

  return fill_layer;
}

IdbFillVia* IdbFillList::add_fill_via(IdbVia* via)
{
  IdbFill* fill = new IdbFill();
  fill->set_type_via();

  IdbFillVia* fill_via = fill->get_via();
  fill_via->set_via(via);

  _fill_list.emplace_back(fill);
  _num_fill++;

  return fill_via;
}

}  // namespace idb
