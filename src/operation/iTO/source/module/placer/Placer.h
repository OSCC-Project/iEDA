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

#include <vector>

#include "../../data/Rects.h"
#include "../../data/RowSpacing.h"
#include "Utility.h"
#include "define.h"

namespace idb {
class IdbRow;
};
namespace ito {

#define toPlacer Placer::get_instance()

class Placer
{
 public:
  static Placer* get_instance();
  static void destroy_instance();

  std::pair<int, int> findNearestSpace(unsigned int master_width, int loc_x, int loc_y);

  void updateRow(unsigned int master_width, int loc_x, int loc_y);

  idb::IdbRow* findRow(int loc_y);

 private:
  static Placer* _instance;

  Placer();
  ~Placer();
  void initRow();

  std::pair<UsedSpace*, int> findNearestSpace(std::vector<UsedSpace*> options, int begin, int end, int dis_margin);

  int placeAlignWithSite(UsedSpace option);

  std::pair<UsedSpace*, int> findNearestRowLegalSpace(int row_idx, unsigned int master_width, int loc_x, int loc_y);

  std::vector<RowSpacing*> _row_space;
  int _row_height;
  int _site_width;
};

}  // namespace ito
