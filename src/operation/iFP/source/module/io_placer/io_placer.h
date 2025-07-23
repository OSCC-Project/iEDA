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

#include <string>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "ifp_enum.h"
#include "ifp_interval.h"

namespace ifp {

struct PadCoordinate
{
  Edge edge;  /// bottom, left, top, right
  idb::IdbOrient orient;
  int begin;  /// coordinate range begin
  int end;    /// coordinate range end
  int coord;  /// the other coordinate value, not change
};

class IoPlacer
{
 public:
  explicit IoPlacer() {}
  ~IoPlacer() {}

  /// operator
  bool autoPlacePins(std::string layer_name, int width, int height, std::vector<std::string> sides);
  bool placePort(std::string pin_name, int32_t x_offset, int32_t y_offset, int32_t rect_width, int32_t rect_height, std::string layer_name);

  bool autoPlacePad(std::vector<std::string> pad_masters = {}, std::vector<std::string> conner_masters = {});
  bool autoIOFiller(std::vector<std::string> filler_name_list = {}, std::string prefix = "IOFIL_");

 private:
  int32_t _iofiller_idx = -1;
  PadCoordinate _pad_coord[4];  /// 4 edge, in order clockwise by bottom, left, top, right

  void set_pad_coords(vector<string> conner_masters = {});
  void placeIOFiller(std::vector<idb::IdbCellMaster*>& fillers, const std::string prefix, PadCoordinate coord);
  void fillInterval(Interval interval, std::vector<idb::IdbCellMaster*> fillers, const std::string prefix, PadCoordinate coord);

  int32_t transUnitDB(double value);
  idb::IdbOrient transferEdgeToOrient(Edge edge);
  bool edgeIsSameToOrient(Edge edge, idb::IdbOrient orient);
  std::string transferOrientToString(idb::IdbOrient orient);
};
}  // namespace ifp