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
#include "DPLayout.hh"

namespace ipl{

DPLayout::DPLayout(int32_t row_num, int32_t region_max_x, int32_t region_max_y): _row_num(row_num), _max_x(region_max_x), _max_y(region_max_y), _dbu(INT32_MIN), _row_height(INT32_MIN), _site_width(INT32_MIN)
{
}

DPLayout::~DPLayout()
{
    // delete row
    for(auto& row_1d_list : _row_2d_list){
        for(auto* row : row_1d_list){
            delete row;
        }
    }
    _row_2d_list.clear();

    // delete interval
    for(auto& interval_1d_list : _interval_2d_list){
        for(auto* interval : interval_1d_list){
            delete interval;
        }
    }
    _interval_2d_list.clear();

    // delete region
    for(auto* region : _region_list){
        delete region;
    }
    _region_list.clear();
    _region_map.clear();

    // delete cell
    for(auto* cell : _cell_list){
        delete cell;
    }
    _cell_list.clear();
    _cell_map.clear();
}

void DPLayout::add_region(DPRegion* region){
    _region_list.push_back(region);
    _region_map.emplace(region->get_name(),region);
}

void DPLayout::add_cell(DPCell* cell){
    _cell_list.push_back(cell);
    _cell_map.emplace(cell->get_name(),cell);
}

DPRegion* DPLayout::find_region(std::string region_name){
    DPRegion* region_ptr = nullptr;
    auto it = _region_map.find(region_name);
    if(it != _region_map.end()){
        region_ptr = it->second;
    }
    return region_ptr;
}

DPCell* DPLayout::find_cell(std::string cell_name){
    DPCell* cell_ptr = nullptr;
    auto it = _cell_map.find(cell_name);
    if(it != _cell_map.end()){
        cell_ptr = it->second;
    }
    return cell_ptr;
}

void DPLayout::resetAllInterval(){
    for(auto& interval_list : _interval_2d_list){
        for(auto* interval : interval_list){
            interval->resetInterval();
        }
    }
}

}