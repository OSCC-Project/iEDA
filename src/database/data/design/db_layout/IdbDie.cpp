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
  _area = 0;
}

IdbDie::~IdbDie()
{
  reset();
}

uint64_t IdbDie::get_area()
{
  if (_area == 0) {
    if (is_polygon()) {
      _area = bg::area(_polygon);
    } else {
      _area = ((uint64_t) get_width()) * ((uint64_t) get_height());
    }
  }

  return _area;
}

uint32_t IdbDie::add_point(IdbCoordinate<int32_t>* pt)
{
  _points.push_back(pt);
  bg::append(_polygon, point_t(pt->get_x(), pt->get_y()));

  if(_points.size() >= RECTANGLE_NUM){
    set_bounding_box();
  }

  return _points.size();
}

bool IdbDie::set_bounding_box()
{
  int32_t llx = INT32_MAX;
  int32_t lly = INT32_MAX;
  int32_t urx = 0;
  int32_t ury = 0;

  for (auto pt : _points) {
    llx = std::min(llx, pt->get_x());
    lly = std::min(lly, pt->get_y());
    urx = std::max(urx, pt->get_x());
    ury = std::max(ury, pt->get_y());
  }

  return IdbObject::set_bounding_box(llx, lly, urx, ury);
}

uint32_t IdbDie::add_point(int32_t x, int32_t y)
{
  IdbCoordinate<int32_t>* point = new IdbCoordinate<int32_t>(x, y);

  return add_point(point);
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
