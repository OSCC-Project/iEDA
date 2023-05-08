/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		IdbDie.h
 * @copyright	(c) 2021 All Rights Reserved.
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
