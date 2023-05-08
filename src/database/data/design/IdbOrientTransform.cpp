/**
 * @project		iDB
 * @file		IdbOrientTransform.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process coodinate transfrom by IdbOrient.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbOrientTransform.h"

#include <algorithm>

namespace idb {

IdbOrientTransform::~IdbOrientTransform()
{
}

bool IdbOrientTransform::transformCoordinate(IdbCoordinate<int32_t>* coordinate_transform)
{
  if (coordinate_transform == nullptr) {
    std::cout << "Error : coordinate is null..." << std::endl;
    return false;
    ;
  }

  switch (_orient) {
    case IdbOrient::kNone:
    case IdbOrient::kMax: {
      break;
    }
    case IdbOrient::kN_R0: {
      transform_r0(coordinate_transform);
      break;
    }
    case IdbOrient::kS_R180: {
      transform_r180(coordinate_transform);
      break;
    }
    case IdbOrient::kW_R90: {
      transform_r90(coordinate_transform);
      break;
    }
    case IdbOrient::kE_R270: {
      transform_r270(coordinate_transform);
      break;
    }
    case IdbOrient::kFN_MY: {
      transform_my(coordinate_transform);
      break;
    }
    case IdbOrient::kFS_MX: {
      transform_mx(coordinate_transform);
      break;
    }
    case IdbOrient::kFW_MX90: {
      transform_mx90(coordinate_transform);
      break;
    }
    case IdbOrient::kFE_MY90: {
      transform_my90(coordinate_transform);
      break;
    }
    default:
      break;
  }

  return true;
}

bool IdbOrientTransform::transformRect(IdbRect* rect)
{
  IdbCoordinate<int32_t> ll = rect->get_low_point();
  IdbCoordinate<int32_t> ur = rect->get_high_point();

  if (!transformCoordinate(&ll) || !transformCoordinate(&ur)) {
    return false;
  }

  int ll_x = std::min(ll.get_x(), ur.get_x());
  int ll_y = std::min(ll.get_y(), ur.get_y());

  int ur_x = std::max(ll.get_x(), ur.get_x());
  int ur_y = std::max(ll.get_y(), ur.get_y());

  rect->set_rect(ll_x, ll_y, ur_x, ur_y);

  return true;
}

// no rotation
void IdbOrientTransform::transform_r0(IdbCoordinate<int32_t>* coordinate_transform)
{
  //   int32_t x = coordinate_transform->get_x() - _original->get_x();
  //   int32_t y = coordinate_transform->get_y() - _original->get_y();

  //   int32_t x_new = -y + _height + _original->get_x();
  //   int32_t y_new = x + _original->get_y();

  //   coordinate_transform->set_xy(x_new, y_new);
  return;
}
/*
 */
void IdbOrientTransform::transform_r90(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = -y + _height + _original->get_x();
  int32_t y_new = x + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_r180(IdbCoordinate<int32_t>* coordinate_transform)
{
  // set _original as coordinate_transform's origin point
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = -x + _width + _original->get_x();
  int32_t y_new = -y + _height + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_r270(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = y + _original->get_x();
  int32_t y_new = -x + _width + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_my(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = -x + _width + _original->get_x();
  int32_t y_new = y + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_mx(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = x + _original->get_x();
  int32_t y_new = -y + _height + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_mx90(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = y + _original->get_x();
  int32_t y_new = x + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

void IdbOrientTransform::transform_my90(IdbCoordinate<int32_t>* coordinate_transform)
{
  int32_t x = coordinate_transform->get_x() - _original->get_x();
  int32_t y = coordinate_transform->get_y() - _original->get_y();

  int32_t x_new = -y + _height + _original->get_x();
  int32_t y_new = -x + _width + _original->get_y();

  coordinate_transform->set_xy(x_new, y_new);
}

}  // namespace idb
