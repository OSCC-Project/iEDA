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
#include "tapcell.h"

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbRow.h"
#include "idm.h"
#include "ifp_enum.h"

namespace ifp {
/**
 * tapcell & endcap
 * distance : the spacing of cell, in dbu
 * tapcell_name : cell name for tapcell
 * endcap_name : cell name for endcap
 */
bool TapCellPlacer::tapCells(double distance, std::string tapcell_name, std::string endcap_name)
{
  auto idb_layout = dmInst->get_idb_layout();

  /// check distance
  auto checkDistance = [](int32_t& inst_space) {
    auto idb_layout = dmInst->get_idb_layout();
    auto core_site = idb_layout->get_sites()->get_core_site();

    if (core_site == nullptr) {
      return false;
    }

    if (inst_space % core_site->get_width() == 0) {
      return true;
    } else {
      int32_t num = inst_space / core_site->get_width();
      inst_space = num * core_site->get_width();

      return false;
    }
  };

  int32_t inst_space = distance * idb_layout->get_units()->get_micron_dbu();
  checkDistance(inst_space);

  /// check cell master
  auto tapcell_master = idb_layout->get_cell_master_list()->find_cell_master(tapcell_name);
  auto endcap_master = idb_layout->get_cell_master_list()->find_cell_master(endcap_name);

  if (tapcell_master == nullptr || endcap_master == nullptr) {
    return false;
  }

  /// create insert cell region
  if (0 == buildTapcellRegion()) {
    return false;
  }

  /// insert cell
  if (insertCell(inst_space, tapcell_name, endcap_name) <= 0) {
    return false;
  }

  return true;
}

/**
 * build tap cell & endcap region in rows
 */
int TapCellPlacer::buildTapcellRegion()
{
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_rows = idb_layout->get_rows();

  auto row_list = idb_rows->get_row_list();
  for (size_t i = 0; i < row_list.size(); i++) {
    buildRegionInRow(row_list[i], i);
    auto y = row_list[i]->get_original_coordinate()->get_y();
    _top_y = std::max(_top_y, y);
    _bottom_y = std::min(_bottom_y, y);
  }

  return _cell_region_list.size();
}
/**
 * build tapcell & endcap region in row by ignore blockage area
 */
void TapCellPlacer::buildRegionInRow(idb::IdbRow* idb_row, int32_t index)
{
  /// row coordinate
  auto row_rect = idb_row->get_bounding_box();
  int32_t idb_row_start_x = row_rect->get_low_x();
  int32_t idb_row_start_y = row_rect->get_low_y();
  int32_t idb_row_end_x = row_rect->get_high_x();
  int32_t idb_row_end_y = row_rect->get_high_y();

  auto idb_design = dmInst->get_idb_design();
  auto idb_blockage_list = idb_design->get_blockage_list();

  std::vector<idb::IdbRect*> rect_list;  /// intersected blockage rect list
  for (auto blockage : idb_blockage_list->get_blockage_list()) {
    if (blockage->is_palcement_blockage()) {
      /// add blocakge shape to region
      for (auto rect : blockage->get_rect_list()) {
        /// row and rect intersected
        if (idb_row_end_y < rect->get_low_y() || idb_row_start_y > rect->get_high_y() || idb_row_start_x > rect->get_low_x()
            || idb_row_end_x < rect->get_high_x()) {
          continue;
        }

        rect_list.emplace_back(rect);
      }
    }
  }

  /// no blockage, save row data
  if (rect_list.size() <= 0) {
    TapcellRegion region;
    region.start = idb_row_start_x;
    region.end = idb_row_end_x;
    region.y = idb_row->get_original_coordinate()->get_y();
    region.orient = idb_row->get_orient();
    region.index = index;

    _cell_region_list.emplace_back(region);

    return;
  } else {
    /// sort blockage rect by coordinate x
    std::sort(rect_list.begin(), rect_list.end(),
              [](idb::IdbRect* rect_1, idb::IdbRect* rect_2) { return rect_1->get_low_x() < rect_2->get_low_x(); });

    /// build region
    for (size_t i = 0; i < rect_list.size(); i++) {
      /// process begin
      if (i == 0) {
        if (idb_row_start_x < rect_list[i]->get_low_x()) {
          TapcellRegion region;
          region.start = idb_row_start_x;
          region.end = rect_list[i]->get_low_x();
          region.y = idb_row->get_original_coordinate()->get_y();
          region.orient = idb_row->get_orient();
          region.index = index;

          _cell_region_list.emplace_back(region);
        }
      }

      /// process end
      if ((i == rect_list.size() - 1)) {
        if (idb_row_end_x > rect_list[i]->get_high_x()) {
          TapcellRegion region;
          region.start = rect_list[i]->get_high_x();
          region.end = idb_row_end_x;
          region.y = idb_row->get_original_coordinate()->get_y();
          region.orient = idb_row->get_orient();
          region.index = index;

          _cell_region_list.emplace_back(region);
        }
      }
      /// process middle region
      if (i > 0 && i < (rect_list.size() - 1)) {
        TapcellRegion region;
        region.start = rect_list[i - 1]->get_high_x();
        region.end = rect_list[i]->get_low_x();
        region.y = idb_row->get_original_coordinate()->get_y();
        region.orient = idb_row->get_orient();
        region.index = index;

        _cell_region_list.emplace_back(region);
      }
    }
  }
}

/**
 * insert tapcell & endcap in region
 */
int TapCellPlacer::insertCell(int32_t inst_space, std::string tapcell_name, std::string endcap_name)
{
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_core = idb_layout->get_core();

  auto tapcell_master = idb_layout->get_cell_master_list()->find_cell_master(tapcell_name);
  auto endcap_master = idb_layout->get_cell_master_list()->find_cell_master(endcap_name);

  int endcap_index = 0;
  int tapcell_index = 0;
  for (auto region : _cell_region_list) {
    /// get width for endcap by orient
    int32_t endcap_width = getCellMasterWidthByOrient(endcap_master, region.orient);

    /// process top and bottom endcap
    if (region.y == _top_y || region.y == _bottom_y) {
      for (auto x = region.start; x < region.end; x += endcap_width) {
        dmInst->createInstance("ENDCAP_" + std::to_string(endcap_index++), endcap_name, x, region.y, region.orient,
                               idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);
      }
    } else {
      /// process area between top and bottom row

      /// insert endcap at the begin
      if ((region.end - region.start) >= endcap_width) {
        dmInst->createInstance("ENDCAP_" + std::to_string(endcap_index++), endcap_name, region.start, region.y, region.orient,
                               idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);
      }
      /// insert endcap at the end
      if ((region.end - region.start) >= (2 * endcap_width)) {
        dmInst->createInstance("ENDCAP_" + std::to_string(endcap_index++), endcap_name, region.end - endcap_width, region.y, region.orient,
                               idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);
      }

      /// insert tapcell in the middle region
      {
        /// get core low bottom x
        int32_t core_start_x = idb_core->get_bounding_box()->get_low_x();

        /// get start & end of this region plus endcap width
        int32_t region_start = region.start + endcap_width;
        int32_t region_end = region.end - endcap_width;

        /// get width for tapcell and endcap by orient
        int32_t tapcell_width = getCellMasterWidthByOrient(tapcell_master, region.orient);
        int32_t coord_x = region_start;
        while ((coord_x + tapcell_width) <= region_end) {
          /// process first tapcell
          if (coord_x == region_start) {
            /// insert tapcell
            if (region.index % 2 == 0) {
              dmInst->createInstance("PHY_" + std::to_string(tapcell_index++), tapcell_name, region_start, region.y, region.orient,
                                     idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);
              /// 以core x为基准，inst_space为间距，奇偶行交错对齐tapcell
              /// 校准偶数行起始点x
              coord_x = core_start_x + ((coord_x - core_start_x) / inst_space + 2) * inst_space;
            } else {
              /// 校准奇数行起始点x
              coord_x = core_start_x + ((coord_x - core_start_x) / inst_space + 1) * inst_space;
            }

            continue;
          }

          /// process middle cell
          dmInst->createInstance("PHY_" + std::to_string(tapcell_index++), tapcell_name, coord_x, region.y, region.orient,
                                 idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);

          /// process last tapcell
          if ((coord_x + 2 * inst_space) >= region_end) {
            /// add tapcell to the end of the row adjacentd to the endcap
            /// at less 2 tapcell width is needed in order to insert tapcell
            if ((region_end - coord_x) > (tapcell_width * 2)) {
              /// insert tapcell at the end,
              /// and coordinate x = region_end - tapcell_width
              if (region.index % 2 == 0) {
                dmInst->createInstance("PHY_" + std::to_string(tapcell_index++), tapcell_name, region_end - tapcell_width, region.y,
                                       region.orient, idb::IdbInstanceType::kDist, idb::IdbPlacementStatus::kFixed);
              }
            }
          }

          coord_x += (inst_space * 2);
        }
      }
    }
  }

  return endcap_index + tapcell_index;
}
/**
 * get width of cell master by orient for a row
 */
int32_t TapCellPlacer::getCellMasterWidthByOrient(idb::IdbCellMaster* cell_master, idb::IdbOrient orinet)
{
  int32_t width = cell_master->get_width();
  int32_t height = cell_master->get_height();

  if (orinet == idb::IdbOrient::kN_R0 || orinet == idb::IdbOrient::kS_R180 || orinet == idb::IdbOrient::kFN_MY
      || orinet == idb::IdbOrient::kFS_MX) {
    return width;
  } else {
    /// rotate 90
    return height;
  }
}

}  // namespace ifp
