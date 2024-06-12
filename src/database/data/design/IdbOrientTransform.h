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
  /// transform to instance coordinate
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

  /// transform to cell master coordinate
  bool cellMasterCoordinate(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_r0(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_r90(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_r180(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_r270(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_my(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_mx(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_mx90(IdbCoordinate<int32_t>* coordinate_transform);
  void cellMaster_my90(IdbCoordinate<int32_t>* coordinate_transform);

 private:
  IdbOrient _orient;
  IdbCoordinate<int32_t>* _original;
  int32_t _width;
  int32_t _height;
};

}  // namespace idb
