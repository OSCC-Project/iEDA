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
/**
 * @File Name: dm_layout.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {

void DataManager::initDie(int ll_x, int ll_y, int ur_x, int ur_y)
{
  IdbDie* die = _layout->get_die();
  die->reset();

  auto io_site = _layout->get_sites()->get_io_site();
  auto corner_site = _layout->get_sites()->get_corner_site();

  if (io_site == nullptr || corner_site == nullptr) {
    die->add_point(ll_x, ll_y);
    die->add_point(ur_x, ur_y);
  } else {
    /// adjust urx & ury
    int corner_width = corner_site->get_width();
    int io_site_width = io_site->get_width();
    int width = ur_x - ll_x;
    int height = ur_y - ll_y;
    ur_x = ll_x + ((width - corner_width * 2) / io_site_width * io_site_width + corner_width * 2);
    ur_y = ll_y + ((height - corner_width * 2) / io_site_width * io_site_width + corner_width * 2);

    die->add_point(ll_x, ll_y);
    die->add_point(ur_x, ur_y);
  }
}

uint64_t DataManager::dieArea()
{
  IdbDie* die = _layout->get_die();

  return die->get_area();
}

double DataManager::dieAreaUm()
{
  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();
  auto* idb_die = _layout->get_die();
  auto die_width = ((double) idb_die->get_width()) / dbu;
  auto die_height = ((double) idb_die->get_height()) / dbu;

  return die_width * die_height;
}

float DataManager::dieUtilization()
{
  uint64_t inst_area = netlistInstArea() + timingInstArea();

  float utilization = ((double) inst_area) / dieAreaUm();

  return utilization;
}

uint64_t DataManager::coreArea()
{
  IdbCore* core = _layout->get_core();

  return core->get_bounding_box()->get_area();
}

double DataManager::coreAreaUm()
{
  int dbu = _design->get_units()->get_micron_dbu() < 0 ? _layout->get_units()->get_micron_dbu() : _design->get_units()->get_micron_dbu();
  auto idb_core_box = _layout->get_core()->get_bounding_box();
  auto core_width = ((double) idb_core_box->get_width()) / dbu;
  auto core_height = ((double) idb_core_box->get_height()) / dbu;

  return core_width * core_height;
}

float DataManager::coreUtilization()
{
  uint64_t inst_area = netlistInstArea() + timingInstArea();

  float utilization = ((double) inst_area) / coreAreaUm();

  return utilization;
}

IdbRow* DataManager::createRow(string row_name, string site_name, int32_t orig_x, int32_t orig_y, IdbOrient site_orient, int32_t num_x,
                               int32_t num_y, int32_t step_x, int32_t step_y)
{
  IdbSites* site_list = _layout->get_sites();
  if (site_list == nullptr) {
    return nullptr;
  }
  IdbSite* site = site_list->find_site(site_name);
  if (site == nullptr) {
    return nullptr;
  }

  IdbRows* row_list_ptr = _layout->get_rows();
  if (row_list_ptr == nullptr) {
    return nullptr;
  }

  return row_list_ptr->createRow(row_name, site, orig_x, orig_y, site_orient, num_x, num_y, step_x, step_y);
}

IdbOrient DataManager::getDefaultOrient(int32_t coord_x, int32_t coord_y)
{
  IdbOrient orient = IdbOrient::kNone;

  IdbRows* row_list_ptr = _layout->get_rows();
  if (row_list_ptr == nullptr) {
    return orient;
  }

  /// find row that contains coordinate x y
  for (auto row : row_list_ptr->get_row_list()) {
    if (row->is_horizontal() && row->get_original_coordinate()->get_y() == coord_y) {
      orient = row->get_orient();
      break;
    } else {
      if (row->get_original_coordinate()->get_x() == coord_x) {
        orient = row->get_orient();
        break;
      }
    }
  }

  return orient;
}

}  // namespace idm
