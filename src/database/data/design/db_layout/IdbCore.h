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
 * @file		IdbCore.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Core information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbObject.h"

namespace idb {

using std::vector;
enum class IdbCoreSide
{
  kNone,
  kLeft,
  kRight,
  kBottom,
  kTop,
  kMax
};

class IdbCore : public IdbObject
{
 public:
  IdbCore() = default;
  ~IdbCore() = default;

  // getter
  IdbCoreSide get_side(IdbCoordinate<int32_t>* coordinate);
  bool is_side_top_or_bottom(IdbCoordinate<int32_t>* coordinate)
  {
    return IdbCoreSide::kTop == get_side(coordinate) || IdbCoreSide::kBottom == get_side(coordinate) ? true : false;
  }
  bool is_side_left_or_right(IdbCoordinate<int32_t>* coordinate)
  {
    return IdbCoreSide::kLeft == get_side(coordinate) || IdbCoreSide::kRight == get_side(coordinate) ? true : false;
  }

  // setter

  // operator

 private:
};

}  // namespace idb
