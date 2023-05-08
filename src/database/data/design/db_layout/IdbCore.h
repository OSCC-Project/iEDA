#pragma once
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
 * @file		IdbCore.h
 * @copyright	(c) 2021 All Rights Reserved.
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
