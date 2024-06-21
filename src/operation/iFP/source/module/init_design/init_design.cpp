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
#include "init_design.h"

#include "IdbDesign.h"
#include "idm.h"

namespace ifp {

int32_t InitDesign::transUnitDB(double value)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  return idb_layout != nullptr ? idb_layout->transUnitDB(value) : -1;
}

bool InitDesign::initDie(double die_lx, double die_ly, double die_ux, double die_uy)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  auto idb_die = idb_layout->get_die();
  idb_die->reset();
  idb_die->add_point(transUnitDB(die_lx), transUnitDB(die_ly));
  idb_die->add_point(transUnitDB(die_ux), transUnitDB(die_uy));

  return true;
}

bool InitDesign::initCore(double core_lx, double core_ly, double core_ux, double core_uy, std::string core_site_name,
                          std::string iocell_site_name, std::string corner_site_name)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  auto idb_die = idb_layout->get_die();
  auto core_site = idb_layout->get_sites()->find_site(core_site_name);
  auto io_site = idb_layout->get_sites()->find_site(iocell_site_name);
  auto corner_site = idb_layout->get_sites()->find_site(corner_site_name);
  if (nullptr == idb_layout || nullptr == idb_die || nullptr == core_site || nullptr == corner_site) {
    return false;
  }

  /// set site
  idb_layout->get_sites()->set_core_site(core_site);
  idb_layout->get_sites()->set_io_site(io_site);
  idb_layout->get_sites()->set_corener_site(corner_site);

  int site_dx = core_site->get_width();
  int site_dy = core_site->get_height();
  // floor core lower left corner to multiple of core_site dx/dy.
  int core_lx_int = (transUnitDB(core_lx) / site_dx) * site_dx;
  int core_ly_int = (transUnitDB(core_ly) / site_dy) * site_dy;
  int core_ux_int = (transUnitDB(core_ux) / site_dx) * site_dx;
  int core_uy_int = (transUnitDB(core_uy) / site_dy) * site_dy;

  /// make enough space for io cell
  //   int32_t io_height = io_site != nullptr ? io_site->get_height() : 0;
  //   if ((core_lx_int - idb_die->get_llx() < io_height) || (core_ly_int - idb_die->get_lly()) < io_height
  //       || (idb_die->get_urx() - core_ux_int) < io_height || (idb_die->get_ury() - core_uy_int) < io_height) {
  //     /// error report, tbd
  //     return false;
  //   }

  /// reset rows
  idb_layout->get_rows()->reset();

  /// create rows
  int site_number = abs(core_ux_int - core_lx_int) / site_dx;  // sites number on one row
  int row_number = abs(core_uy_int - core_ly_int) / site_dy;   // row number
  int index_y = core_ly_int;
  for (int row = 0; row < row_number; row++) {
    auto orient = row % 2 == 0 ? idb::IdbOrient::kFS_MX : idb::IdbOrient::kN_R0;

    /// set original horizontal
    dmInst->createRow(("ROW_" + std::to_string(row)), core_site_name, core_lx_int, index_y, orient, site_number, 1, site_dx, 0);

    index_y += site_dy;
  }

  /// set core boundary
  auto idb_core = idb_layout->get_core();
  idb_core->set_bounding_box(core_lx_int, core_ly_int, core_ux_int, core_uy_int);

  return true;
}

/**
 * @brief generate tracks
 *
 * @param layer_name
 * @param x_offset
 * @param x_pitch
 * @param y_offset
 * @param y_pitch
 * @return true
 * @return false
 */
bool InitDesign::makeTracks(std::string layer_name, int x_offset, int x_pitch, int y_offset, int y_pitch)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  auto* routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layout->get_layers()->find_layer(layer_name));
  if (routing_layer == nullptr) {
    std::cout << "[FpApi error] : routing_layer " << layer_name << " do not exist!" << std::endl;
    return false;
  }

  auto track_grid_x = idb_layout->get_track_grid_list()->add_track_grid();
  track_grid_x->add_layer_list(routing_layer);
  routing_layer->add_track_grid(track_grid_x);
  track_grid_x->get_track()->set_direction(idb::IdbTrackDirection::kDirectionX);
  track_grid_x->get_track()->set_pitch(x_pitch);
  track_grid_x->get_track()->set_start(x_offset);
  int track_number = (int) ((idb_layout->get_die()->get_width() - x_offset) / x_pitch);
  track_grid_x->set_track_number(track_number);

  auto track_grid_y = idb_layout->get_track_grid_list()->add_track_grid();
  track_grid_y->add_layer_list(routing_layer);
  routing_layer->add_track_grid(track_grid_y);
  track_grid_y->get_track()->set_direction(idb::IdbTrackDirection::kDirectionY);
  track_grid_y->get_track()->set_pitch(y_pitch);
  track_grid_y->get_track()->set_start(y_offset);
  track_number = (int) ((idb_layout->get_die()->get_height() - y_offset) / y_pitch);
  track_grid_y->set_track_number(track_number);

  return true;
}

}  // namespace ifp