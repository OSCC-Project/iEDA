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

namespace idb {
class IdbDesign;
}

namespace ifp {

class InitDesign
{
 public:
  explicit InitDesign() {}
  ~InitDesign() {}

  /// operator
  bool initDie(double die_lx, double die_ly, double die_ux, double die_uy);
  bool initCore(double core_lx, double core_ly, double core_ux, double core_uy, std::string core_site_name, std::string iocell_site_name,
                std::string corner_site_name);
  bool makeTracks(std::string layer_name, int x_offset, int x_pitch, int y_offset, int y_pitch);

 private:
  int32_t transUnitDB(double value);

  ///

 public:
  /*****************************************************************/
  //   void initFloorplan(double die_lx, double die_ly, double die_ux, double die_uy, double core_lx, double core_ly, double core_ux,
  //                      double core_uy, string core_site_name, string iocell_site_name);

  //   void autoPlacePins(string pin_layer_name, FloorplanDb* db);
  //   void make_tracks();
  //   void make_layer_tracks(int layer, int x_offset, int x_pitch, int y_offset, int y_pitch);
  //   void set_floorplandb(FloorplanDb* floorplandb) { _floorplandb = floorplandb; }

 protected:
  //   int32_t designArea();
  //   int32_t metersToMfgGrid(double dist);

  //   void make_layer_tracks(IdbLayerRouting* routing_layer);

  //   void make_core_rows(IdbSite* site, int core_lx, int core_ly, int core_ux, int core_uy);
  //   IdbRow* createRow(string name, IdbSite* site, int32_t origin_x, int32_t origin_y, IdbOrient site_orient, int32_t num, int32_t step,
  //                     bool is_horizontal);
  //   void updateVoltageDomain(IdbSite* site, int core_lx, int core_ly, int core_ux, int core_uy);
  //   void autoPlacePins(IdbLayer* pin_layer, IdbRect& core);
};
}  // namespace ifp