/**
 * @project		iDB
 * @file		IdbObject.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Core information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbObject.h"

namespace idb {

IdbObject::IdbObject()
{
  _bounding_box = new IdbRect();
  _id = 0;
}

IdbObject::~IdbObject()
{
  if (_bounding_box != nullptr) {
    delete _bounding_box;
    _bounding_box = nullptr;
  }
}

bool IdbObject::set_bounding_box(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  if (_bounding_box != nullptr) {
    _bounding_box->set_rect(ll_x, ll_y, ur_x, ur_y);
    return true;
  }

  return false;
}

void IdbObject::set_bounding_box(IdbRect bounding_box)
{
  set_bounding_box(bounding_box.get_low_x(), bounding_box.get_low_y(), bounding_box.get_high_x(), bounding_box.get_high_y());
}

}  // namespace idb
