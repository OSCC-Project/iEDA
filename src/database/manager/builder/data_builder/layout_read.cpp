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
 * @file		layout_read.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a data builder to build data structure by read layout data file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "layout_read.h"

namespace idb {
LayoutRead::LayoutRead()
{
  _layout = new IdbLayout();
}

IdbLayout* LayoutRead::readLayout(const char* folder)
{
  _folder = folder;

  load_manufacture_grid();
  load_units();
  load_die();
  load_core();
  load_layers();
  load_sites();
  load_rows();
  load_gcell_grid_list();
  load_track_grid_list();
  load_cell_master_list();
  load_via_list();
  load_via_rule_list();

  return _layout;
}

int32_t LayoutRead::load_manufacture_grid()
{
  string file_path = _folder;
  file_path.append("/manufacture_grid.idb");

  int32_t manufacture_grid = 0;
  shared_ptr<IdbManufactureGridHeader> header
      = make_shared<IdbManufactureGridHeader>(IdbFileHeaderType::kManufactureGrid, file_path.c_str(), &manufacture_grid);

  (*header).load_header();
  (*header).load_data();

  _layout->set_manufacture_grid(manufacture_grid);

  return kDbSuccess;
}

IdbUnits* LayoutRead::load_units()
{
  string file_path = _folder;
  file_path.append("/units.idb");

  shared_ptr<IdbUnitsHeader> header = make_shared<IdbUnitsHeader>(IdbFileHeaderType::kUnits, file_path.c_str(), _layout->get_units());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbDie* LayoutRead::load_die()
{
  string file_path = _folder;
  file_path.append("/die.idb");

  shared_ptr<IdbDieHeader> header = make_shared<IdbDieHeader>(IdbFileHeaderType::kDie, file_path.c_str(), _layout->get_die());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbLayers* LayoutRead::load_layers()
{
  string file_path = _folder;
  file_path.append("/layers.idb");

  shared_ptr<IdbLayersHeader> header = make_shared<IdbLayersHeader>(IdbFileHeaderType::kLayers, file_path.c_str(), _layout->get_layers());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbCore* LayoutRead::load_core()
{
  string file_path = _folder;
  file_path.append("/core.idb");

  shared_ptr<IdbCoreHeader> header = make_shared<IdbCoreHeader>(IdbFileHeaderType::kCore, file_path.c_str(), _layout->get_core());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbSites* LayoutRead::load_sites()
{
  string file_path = _folder;
  file_path.append("/sites.idb");

  shared_ptr<IdbSitesHeader> header = make_shared<IdbSitesHeader>(IdbFileHeaderType::kSites, file_path.c_str(), _layout->get_sites());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbRows* LayoutRead::load_rows()
{
  string file_path = _folder;
  file_path.append("/rows.idb");

  shared_ptr<IdbRowsHeader> header
      = make_shared<IdbRowsHeader>(IdbFileHeaderType::kRows, file_path.c_str(), _layout->get_rows(), _layout->get_sites());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbGCellGridList* LayoutRead::load_gcell_grid_list()
{
  string file_path = _folder;
  file_path.append("/gcell_grid_list.idb");

  shared_ptr<IdbGcellGridHeader> header
      = make_shared<IdbGcellGridHeader>(IdbFileHeaderType::kGCellGridList, file_path.c_str(), _layout->get_gcell_grid_list());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbTrackGridList* LayoutRead::load_track_grid_list()
{
  string file_path = _folder;
  file_path.append("/track_grid_list.idb");

  shared_ptr<IdbTrackGridHeader> header = make_shared<IdbTrackGridHeader>(IdbFileHeaderType::kTrackGridList, file_path.c_str(),
                                                                          _layout->get_track_grid_list(), _layout->get_layers());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbCellMasterList* LayoutRead::load_cell_master_list()
{
  string file_path = _folder;
  file_path.append("/cell_master_list.idb");

  shared_ptr<IdbCellMasterHeader> header = make_shared<IdbCellMasterHeader>(IdbFileHeaderType::kCellMasterList, file_path.c_str(),
                                                                            _layout->get_cell_master_list(), _layout->get_layers());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbVias* LayoutRead::load_via_list()
{
  string file_path = _folder;
  file_path.append("/via_list.idb");

  shared_ptr<IdbViaListHeader> header
      = make_shared<IdbViaListHeader>(IdbFileHeaderType::kVias, file_path.c_str(), _layout->get_via_list(), _layout->get_layers());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

IdbViaRuleList* LayoutRead::load_via_rule_list()
{
  string file_path = _folder;
  file_path.append("/via_rule_list.idb");

  shared_ptr<IdbViaRuleListHeader> header = make_shared<IdbViaRuleListHeader>(IdbFileHeaderType::kViaRuleList, file_path.c_str(),
                                                                              _layout->get_via_rule_list(), _layout->get_layers());

  (*header).load_header();
  (*header).load_data();

  return kDbSuccess;
}

}  // namespace idb
