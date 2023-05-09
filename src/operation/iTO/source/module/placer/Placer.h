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

#include "Rect.h"
#include "RowSpacing.h"
#include "Utility.h"
#include "ids.hpp"

using std::pair;

namespace ito {
using namespace idb;

class Placer {
 public:
  Placer(IdbBuilder *idb) : _idb_builder(idb) { initRow(); }
  ~Placer();

  pair<int, int> findNearestSpace(unsigned int master_width, int loc_x, int loc_y);

  void updateRow(unsigned int master_width, int loc_x, int loc_y);

  IdbRow *findRow(int loc_y);

 private:
  void initRow();

  pair<UsedSpace *, int> findNearestSpace(vector<UsedSpace *> options, int begin, int end,
                                          int dis_margin);

  int placeAlignWithSite(UsedSpace option);

  pair<UsedSpace *, int> findNearestRowLegalSpace(int row_idx, unsigned int master_width,
                                                  int loc_x, int loc_y) {
    vector<UsedSpace *> alter_option =
        _row_space[row_idx]->searchFeasiblePlace(loc_x, loc_x + master_width, 3);
    int row_height1 = (row_idx * _row_height) + _core.get_y_min();
    // The distance needed to move to the row
    int                    dis_margin1 = abs(row_height1 - loc_y);
    pair<UsedSpace *, int> opt =
        findNearestSpace(alter_option, loc_x, loc_x + master_width, dis_margin1);
    return opt;
  }
  /* data */
  IdbBuilder    *_idb_builder;
  ito::Rectangle _core;

  vector<RowSpacing *> _row_space;
  int                  _row_height;
  int                  _site_width;
};

} // namespace ito
