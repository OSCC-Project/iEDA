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

#include <stdint.h>
#include <limits.h>

#include <iostream>
#include <string>
#include <vector>

using std::pair;
using std::string;
using std::vector;

namespace idb {
enum class IdbOrient : uint8_t;
class IdbCellMaster;
class IdbRow;
}  // namespace idb
namespace ifp {
struct TapcellRegion
{
  int32_t index;
  int32_t start;
  int32_t end;
  int32_t y;
  idb::IdbOrient orient;
};

class TapCellPlacer
{
 public:
  TapCellPlacer() = default;
  ~TapCellPlacer() = default;

  bool tapCells(double distance, std::string tapcell_name, std::string endcap_name);

 private:
  std::vector<TapcellRegion> _cell_region_list;
  int _top_y = INT_MIN;
  int _bottom_y = INT_MAX;

  bool checkDistance(int32_t distance);
  int buildTapcellRegion();
  void buildRegionInRow(idb::IdbRow* idb_row, int32_t index);

  int insertCell(int32_t inst_space, std::string tapcell_name, std::string endcap_name);
  int32_t getCellMasterWidthByOrient(idb::IdbCellMaster* cell_master, idb::IdbOrient orinet);
};

}  // namespace ifp