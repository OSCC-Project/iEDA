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
 * @Date: 2023-03-06 14:15:18
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-06 14:25:08
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/config/DetailPlacerConfig.hh
 * @Description: DP Config
 * 
 * 
 */
#ifndef IPL_DPCONFIG_H
#define IPL_DPCONFIG_H

#include <string>

namespace ipl {

class DPConfig{
public:
    DPConfig() = default;
    DPConfig(const DPConfig&) = default;
    DPConfig(DPConfig&&) = default;
    ~DPConfig() = default;

    DPConfig& operator=(const DPConfig&) = default;
    DPConfig& operator=(DPConfig&&) = default;

    // getter
    int32_t get_thread_num() const { return _thread_num;}
    int32_t get_max_displacement() const { return _max_displacement;}
    int32_t get_global_padding() const { return _global_padding;}
    int32_t get_grid_cnt_x() const { return _grid_cnt_x;}
    int32_t get_grid_cnt_y() const { return _grid_cnt_y;}
    int32_t isEnableNetworkflow() const { return _enable_networkflow;} 

    // setter
    void set_thread_num(int32_t num_thread) { _thread_num = num_thread;}
    void set_max_displacement(int32_t max_displacement) { _max_displacement = max_displacement;}
    void set_global_padding(int32_t padding) { _global_padding = padding;}
    void set_grid_cnt_x(int32_t grid_cnt_x) { _grid_cnt_x = grid_cnt_x;}
    void set_grid_cnt_y(int32_t grid_cnt_y) { _grid_cnt_y = grid_cnt_y;}
    void set_enable_networkflow(int32_t enable_networkflow) {_enable_networkflow = enable_networkflow;}

private:
    int32_t _thread_num;
    int32_t _max_displacement;
    int32_t _global_padding;
    int32_t _enable_networkflow;

    // tmp keep the same as global placement
    int32_t _grid_cnt_x;
    int32_t _grid_cnt_y;
};

}


#endif
