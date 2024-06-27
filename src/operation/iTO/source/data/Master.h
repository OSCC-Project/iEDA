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

#include "IdbCellMaster.h"
namespace ito {

class Master
{
 public:
  Master(idb::IdbCellMaster* idb_master)
  {
    _width = idb_master->get_width();
    _height = idb_master->get_height();
  }

  ~Master() = default;

  // bool isAutoPlaceable() { return _is_auto_placeable; }

  unsigned int get_width() const { return _width; }
  unsigned int get_height() const { return _height; }

 private:
  unsigned int _width;
  unsigned int _height;

  // bool _is_auto_placeable = false;
};

}  // namespace ito