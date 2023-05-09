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
/**
 * @project		iDB
 * @file		layout_write.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


        There is a data builder to write data file from data structure.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "../../def_builder/def_service/def_service.h"
#include "header.h"

namespace idb {

using std::string;
using std::vector;

class LayoutWrite
{
 public:
  LayoutWrite(IdbLayout* layout);
  ~LayoutWrite();

  // getter
  IdbLayout* get_layout() { return _layout; };

  // writer
  int32_t save_manufacture_grid();
  int32_t save_units();
  int32_t save_die();
  int32_t save_core();
  int32_t save_layers();
  int32_t save_sites();
  int32_t save_rows();
  int32_t save_gcell_grid_list();
  int32_t save_track_grid_list();
  int32_t save_cell_master_list();
  int32_t save_via_list();
  int32_t save_via_rule_list();

  void set_start_time(clock_t time) { _start_time = time; }
  void set_end_time(clock_t time) { _end_time = time; }
  float time_eclips() { return (float(_end_time - _start_time)) / CLOCKS_PER_MS; }

  // operator
  bool writeLayout(const char* folder);

 private:
  IdbLayout* _layout = nullptr;
  const char* _folder;
  int32_t _index = 0;
  clock_t _start_time = 0;
  clock_t _end_time = 0;
};
}  // namespace idb
