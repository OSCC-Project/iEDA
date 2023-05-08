#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		layout_read.h
 * @author		Yell
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


    There is a data builder to build data structure by read layout data file.
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

  class LayoutRead {
   public:
    LayoutRead();
    ~LayoutRead() = default;

    // reader
    int32_t load_manufacture_grid();
    IdbUnits* load_units();
    IdbDie* load_die();
    IdbCore* load_core();
    IdbLayers* load_layers();
    IdbSites* load_sites();
    IdbRows* load_rows();
    IdbGCellGridList* load_gcell_grid_list();
    IdbTrackGridList* load_track_grid_list();
    IdbCellMasterList* load_cell_master_list();
    IdbVias* load_via_list();
    IdbViaRuleList* load_via_rule_list();

    void set_start_time(clock_t time) { _start_time = time; }
    void set_end_time(clock_t time) { _end_time = time; }
    float time_eclips() { return (float(_end_time - _start_time)) / CLOCKS_PER_MS; }

    // operator
    IdbLayout* readLayout(const char* folder);

   private:
    IdbLayout* _layout = nullptr;
    const char* _folder;
    int32_t _index      = 0;
    clock_t _start_time = 0;
    clock_t _end_time   = 0;
  };
}  // namespace idb
