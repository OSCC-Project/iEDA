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

#include "BoostType.h"
#include "DrcEdge.h"
#include "DrcEnum.h"

namespace idrc {
class DrcCorner : public BoostPoint
{
 public:
  DrcCorner() : BoostPoint(), _corner_dir(CornerDirEnum::kNone) {}

  DrcCorner* getPrevCorner() const { return _pre_corner; }
  DrcCorner* getNextCorner() const { return _next_corner; }
  DrcEdge* getPrevEdge() const { return _pre_edge; }
  DrcEdge* getNextEdge() const { return _next_edge; }
  // BoostPoint getPoint() const { return _point; }
  // frCornerTypeEnum getType() const { return cornerType; }
  CornerDirEnum getDir() const { return _corner_dir; }
  bool isFixed() const { return _fixed; }

  // setters
  void setPrevCorner(DrcCorner* in) { _pre_corner = in; }
  void setNextCorner(DrcCorner* in) { _next_corner = in; }
  void setPrevEdge(DrcEdge* in) { _pre_edge = in; }
  void setNextEdge(DrcEdge* in) { _next_edge = in; }
  // void setType(frCornerTypeEnum in) { cornerType = in; }
  void setDir(CornerDirEnum in) { _corner_dir = in; }
  void setFixed(bool in) { _fixed = in; }

 private:
  // BoostPoint _point;
  DrcCorner* _pre_corner;
  DrcCorner* _next_corner;
  DrcEdge* _pre_edge;
  DrcEdge* _next_edge;
  // DrcCornerTypeEnum _corner_type;
  CornerDirEnum _corner_dir;  // points away from poly for convex and concave
  bool _fixed;
};
}  // namespace idrc