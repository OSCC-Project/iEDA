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
 * @LastEditTime: 2023-02-16 16:43:17
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGLayout.cc
 * @Description: 
 * 
 * 
 */
#include "LGLayout.hh"

namespace ipl{

LGLayout::LGLayout(int32_t row_num, int32_t region_max_x, int32_t region_max_y): _row_num(row_num), _dbu(0), _max_x(region_max_x), _max_y(region_max_y){}

LGLayout::~LGLayout()
{
    // delete LGRow*
    for(size_t i=0; i<_row_2d_list.size(); i++){
        for(auto* row : _row_2d_list.at(i)){
            delete row;
        }
    }
    _row_2d_list.clear();

    // delete LGInterval*
    for(size_t i=0; i<_interval_2d_list.size(); i++){
        for(auto* segment : _interval_2d_list.at(i)){
            delete segment;
        }
    }
    _interval_2d_list.clear();

    // delete LGRegion*
    for(auto* region : _region_list){
        delete region;
    }
    _region_list.clear();
    _region_map.clear();

    // delete LGCell*
    for(auto* cell : _cell_list){
        delete cell;
    }
    _cell_list.clear();
    _cell_map.clear();
}

int32_t LGLayout::get_row_height(){
    return _row_2d_list.at(0).at(0)->get_site()->get_height();
}

int32_t LGLayout::get_site_width(){
    return _row_2d_list.at(0).at(0)->get_site()->get_width();
}

void LGLayout::add_region(LGRegion* region) 
{
    _region_list.push_back(region);
    _region_map.emplace(region->get_name(),region);
}

void LGLayout::add_cell(LGCell* cell)
{
    _cell_list.push_back(cell);
    _cell_map.emplace(cell->get_name(),cell);
}

LGRegion* LGLayout::find_region(std::string region_name){
    LGRegion* region = nullptr;
    auto iter = _region_map.find(region_name);
    if(iter != _region_map.end()){
        region = iter->second;
    }
    return region;
}

LGCell* LGLayout::find_cell(std::string cell_name){
    LGCell* cell = nullptr;
    auto iter = _cell_map.find(cell_name);
    if(iter != _cell_map.end()){
        cell = iter->second;
    }
    return cell;
}

LGInterval* LGLayout::find_interval(std::string interval_name){
    std::string row_id_str, column_id_str;

    size_t underscore_pos = interval_name.find('_');
    row_id_str = interval_name.substr(0, underscore_pos);
    column_id_str = interval_name.substr(underscore_pos + 1);

    int row_id = std::stoi(row_id_str);
    int column_id = std::stoi(column_id_str);

    return _interval_2d_list[row_id][column_id];
}

}