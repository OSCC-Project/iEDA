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
 * @Date: 2023-03-03 19:03:42
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-06 19:27:05
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPLayout.hh
 * @Description: Layout Infomation of detail placement
 * 
 * 
 */
#ifndef IPL_DPLAYOUT_H
#define IPL_DPLAYOUT_H

#include <string>
#include <vector>
#include <map>

#include "DPRow.hh"
#include "DPInterval.hh"
#include "DPRegion.hh"
#include "DPCell.hh"
#include "DPBin.hh"

namespace ipl {
class DPLayout
{
public:
    DPLayout() = delete;
    DPLayout(int32_t row_num, int32_t region_max_x, int32_t region_max_y);
    DPLayout(const DPLayout&) = delete;
    DPLayout(DPLayout&&) = delete;
    ~DPLayout();

    DPLayout& operator=(const DPLayout&) = delete;
    DPLayout& operator=(DPLayout&&) = delete;

    // getter
    int32_t get_row_num() const { return _row_num;}
    int32_t get_dbu() const { return _dbu;}
    int32_t get_max_x() const { return _max_x;}
    int32_t get_max_y() const { return _max_y;}
    const std::vector<std::vector<DPRow*>>& get_row_2d_list() { return _row_2d_list;}
    const std::vector<std::vector<DPInterval*>>& get_interval_2d_list() { return _interval_2d_list;}
    const std::vector<DPRegion*>& get_region_list() { return _region_list;}
    const std::vector<DPCell*>& get_cell_list() { return _cell_list;}
    int32_t get_row_height() const { return _row_height;}
    int32_t get_site_width() const { return _site_width;}

    // setter
    void set_dbu(int32_t dbu) { _dbu = dbu;}
    void set_row_height(int32_t row_height) { _row_height = row_height;}
    void set_site_width(int32_t site_width) { _site_width = site_width;}
    void set_row_2d_list(std::vector<std::vector<DPRow*>>& row_2d_list) {_row_2d_list = std::move(row_2d_list);}
    void set_interval_2d_list(std::vector<std::vector<DPInterval*>>& interal_2d_list) { _interval_2d_list = std::move(interal_2d_list);}
    void add_region(DPRegion* region);
    void add_cell(DPCell* cell);

    // function
    DPRegion* find_region(std::string region_name);
    DPCell* find_cell(std::string cell_name);

    void resetAllInterval();

private:
    int32_t _row_num;
    int32_t _max_x;
    int32_t _max_y;
    int32_t _dbu;
    int32_t _row_height;
    int32_t _site_width;
    std::vector<std::vector<DPRow*>> _row_2d_list;
    std::vector<std::vector<DPInterval*>> _interval_2d_list;
    std::vector<DPRegion*> _region_list;
    std::vector<DPCell*> _cell_list;

    std::map<std::string, DPRegion*> _region_map;
    std::map<std::string, DPCell*> _cell_map;

};
}
#endif