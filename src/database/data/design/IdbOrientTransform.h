#pragma once
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

#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "IdbEnum.h"

namespace idb {

class IdbOrientTransform
{
 public:
  explicit IdbOrientTransform(IdbOrient orient, IdbCoordinate<int32_t>* original, int32_t width, int32_t height)
  {
    _orient = orient;
    _original = original;
    _width = width;
    _height = height;
  }
  ~IdbOrientTransform();

  // getter

  // setter

  // Operator
  bool transformCoordinate(IdbCoordinate<int32_t>* coordinate_transform);
  bool transformRect(IdbRect* rect);
  void transform_r0(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_r90(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_r180(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_r270(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_my(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_mx(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_mx90(IdbCoordinate<int32_t>* coordinate_transform);
  void transform_my90(IdbCoordinate<int32_t>* coordinate_transform);

 private:
  IdbOrient _orient;
  IdbCoordinate<int32_t>* _original;
  int32_t _width;
  int32_t _height;
};

}  // namespace idb
