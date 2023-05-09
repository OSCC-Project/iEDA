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
 * @file		IdbDie.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Die Area information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbDie.h"

namespace idb {

IdbDie::IdbDie()
{
  _width = -1;
  _height = -1;
}

IdbDie::~IdbDie()
{
  reset();
}

void IdbDie::set_points(vector<IdbCoordinate<int32_t>*> points)
{
  _points = std::move(points);

  if (_points.size() >= kMaxPointsNumber) {
    _width = get_urx() - get_llx();
    _height = get_ury() - get_lly();

    set_bounding_box();
  }
}

uint32_t IdbDie::add_point(IdbCoordinate<int32_t>* pt)
{
  _points.push_back(pt);

  //<!--------------tbd--------------
  //<!---only support rectangle------
  //_points[0] & _points[1] construct rectangle
  if (_points.size() >= kMaxPointsNumber) {
    _width = get_urx() - get_llx();
    _height = get_ury() - get_lly();

    set_bounding_box();
  }

  return _points.size();
}

uint32_t IdbDie::add_point(int32_t x, int32_t y)
{
  IdbCoordinate<int32_t>* point = new IdbCoordinate<int32_t>(x, y);

  return add_point(point);
}

bool IdbDie::set_bounding_box()
{
  return IdbObject::set_bounding_box(get_llx(), get_lly(), get_urx(), get_ury());
}

void IdbDie::print()
{
  vector<IdbCoordinate<int32_t>*>::iterator it = _points.begin();
  for (; it != _points.end(); ++it) {
    IdbCoordinate<int32_t>* point = *it;
    std::cout << point->get_x() << "  " << point->get_y() << std::endl;
  }
}

}  // namespace idb
