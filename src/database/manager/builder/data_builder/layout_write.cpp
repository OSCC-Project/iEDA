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
 * @project		iDB
 * @file		layout_write.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a layout builder to write binary layout file from data structure.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "layout_write.h"

namespace idb {
LayoutWrite::LayoutWrite(IdbLayout* layout)
{
  _layout = layout;
}

LayoutWrite::~LayoutWrite()
{
}

bool LayoutWrite::writeLayout(const char* folder)
{
  _folder = folder;
  save_manufacture_grid();
  save_units();
  save_die();
  save_core();
  save_layers();
  save_sites();
  save_rows();
  save_gcell_grid_list();
  save_track_grid_list();
  save_cell_master_list();
  save_via_list();
  save_via_rule_list();

  return kDbSuccess;
}

int32_t LayoutWrite::save_manufacture_grid()
{
  string file_path = _folder;
  file_path.append("/manufacture_grid.idb");

  int32_t manufacture_grid = _layout->get_munufacture_grid();

  shared_ptr<IdbManufactureGridHeader> header
      = make_shared<IdbManufactureGridHeader>(IdbFileHeaderType::kManufactureGrid, file_path.c_str(), &manufacture_grid);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_units()
{
  string file_path = _folder;
  file_path.append("/units.idb");

  IdbUnits* units = _layout->get_units();

  shared_ptr<IdbUnitsHeader> header = make_shared<IdbUnitsHeader>(IdbFileHeaderType::kUnits, file_path.c_str(), units);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_die()
{
  string file_path = _folder;
  file_path.append("/die.idb");

  IdbDie* die = _layout->get_die();

  shared_ptr<IdbDieHeader> header = make_shared<IdbDieHeader>(IdbFileHeaderType::kDie, file_path.c_str(), die);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_layers()
{
  string file_path = _folder;
  file_path.append("/layers.idb");

  IdbLayers* layers = _layout->get_layers();

  shared_ptr<IdbLayersHeader> header = make_shared<IdbLayersHeader>(IdbFileHeaderType::kLayers, file_path.c_str(), layers);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_core()
{
  string file_path = _folder;
  file_path.append("/core.idb");

  IdbCore* core = _layout->get_core();

  shared_ptr<IdbCoreHeader> header = make_shared<IdbCoreHeader>(IdbFileHeaderType::kCore, file_path.c_str(), core);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_sites()
{
  string file_path = _folder;
  file_path.append("/sites.idb");

  IdbSites* sites = _layout->get_sites();

  shared_ptr<IdbSitesHeader> header = make_shared<IdbSitesHeader>(IdbFileHeaderType::kSites, file_path.c_str(), sites);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_rows()
{
  string file_path = _folder;
  file_path.append("/rows.idb");

  IdbRows* rows = _layout->get_rows();

  shared_ptr<IdbRowsHeader> header = make_shared<IdbRowsHeader>(IdbFileHeaderType::kRows, file_path.c_str(), rows, nullptr);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_gcell_grid_list()
{
  string file_path = _folder;
  file_path.append("/gcell_grid_list.idb");

  IdbGCellGridList* gcell_grid_list = _layout->get_gcell_grid_list();

  shared_ptr<IdbGcellGridHeader> header
      = make_shared<IdbGcellGridHeader>(IdbFileHeaderType::kGCellGridList, file_path.c_str(), gcell_grid_list);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_track_grid_list()
{
  string file_path = _folder;
  file_path.append("/track_grid_list.idb");

  IdbTrackGridList* track_grid_list = _layout->get_track_grid_list();

  shared_ptr<IdbTrackGridHeader> header
      = make_shared<IdbTrackGridHeader>(IdbFileHeaderType::kTrackGridList, file_path.c_str(), track_grid_list, nullptr);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_cell_master_list()
{
  string file_path = _folder;
  file_path.append("/cell_master_list.idb");

  IdbCellMasterList* cell_master_list = _layout->get_cell_master_list();

  shared_ptr<IdbCellMasterHeader> header
      = make_shared<IdbCellMasterHeader>(IdbFileHeaderType::kCellMasterList, file_path.c_str(), cell_master_list, nullptr);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_via_list()
{
  string file_path = _folder;
  file_path.append("/via_list.idb");

  IdbVias* via_list = _layout->get_via_list();

  shared_ptr<IdbViaListHeader> header = make_shared<IdbViaListHeader>(IdbFileHeaderType::kVias, file_path.c_str(), via_list, nullptr);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

int32_t LayoutWrite::save_via_rule_list()
{
  string file_path = _folder;
  file_path.append("/via_rule_list.idb");

  IdbViaRuleList* via_rule_list = _layout->get_via_rule_list();

  shared_ptr<IdbViaRuleListHeader> header
      = make_shared<IdbViaRuleListHeader>(IdbFileHeaderType::kViaRuleList, file_path.c_str(), via_rule_list, nullptr);

  (*header).save_header();
  (*header).save_data();

  return kDbSuccess;
}

}  // namespace idb
