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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-07 11:00:35
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-16 16:43:00
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGLayout.hh
 * @Description: Layout manager of LG
 *
 *
 */
#ifndef IPL_LGLAYOUT_H
#define IPL_LGLAYOUT_H

#include <map>
#include <string>
#include <vector>

#include "LGCell.hh"
#include "LGInterval.hh"
#include "LGRegion.hh"
#include "LGRow.hh"

namespace ipl {
class LGLayout
{
 public:
  LGLayout() = delete;
  LGLayout(int32_t row_num, int32_t region_max_x, int32_t region_max_y);
  LGLayout(const LGLayout&) = delete;
  LGLayout(LGLayout&&) = delete;
  ~LGLayout();

  LGLayout& operator=(const LGLayout&) = delete;
  LGLayout& operator=(LGLayout&&) = delete;

  // getter
  int32_t get_row_num() const { return _row_num; }
  int32_t get_dbu() const { return _dbu; }
  int32_t get_max_x() const { return _max_x; }
  int32_t get_max_y() const { return _max_y; }
  std::vector<std::vector<LGRow*>>& get_row_2d_list() { return _row_2d_list; }
  std::vector<std::vector<LGInterval*>>& get_interval_2d_list() { return _interval_2d_list; }
  std::vector<LGRegion*>& get_region_list() { return _region_list; }
  std::vector<LGCell*>& get_cell_list() { return _cell_list; }
  int32_t get_row_height();
  int32_t get_site_width();

  // setter
  void set_dbu(int32_t dbu) { _dbu = dbu; }
  void set_row_2d_list(std::vector<std::vector<LGRow*>> row_2d_list) { _row_2d_list = std::move(row_2d_list); }
  void set_interval_2d_list(std::vector<std::vector<LGInterval*>> segment_2d_list) { _interval_2d_list = std::move(segment_2d_list); }
  void add_region(LGRegion* region);
  void add_cell(LGCell* cell);

  // function
  LGRegion* find_region(std::string region_name);
  LGCell* find_cell(std::string cell_name);
  LGInterval* find_interval(std::string interval_name);

 private:
  int32_t _row_num;
  int32_t _dbu;
  int32_t _max_x;
  int32_t _max_y;
  std::vector<std::vector<LGRow*>> _row_2d_list;
  std::vector<std::vector<LGInterval*>> _interval_2d_list;
  std::vector<LGRegion*> _region_list;
  std::vector<LGCell*> _cell_list;

  std::map<std::string, LGRegion*> _region_map;
  std::map<std::string, LGCell*> _cell_map;
};
}  // namespace ipl
#endif