/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-06 14:15:18
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-06 14:25:08
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/config/DetailPlacerConfig.hh
 * @Description: DP Config
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
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

    // setter
    void set_thread_num(int32_t num_thread) { _thread_num = num_thread;}
    void set_max_displacement(int32_t max_displacement) { _max_displacement = max_displacement;}
    void set_global_padding(int32_t padding) { _global_padding = padding;}

private:
    int32_t _thread_num;
    int32_t _max_displacement;
    int32_t _global_padding;
};

}


#endif
