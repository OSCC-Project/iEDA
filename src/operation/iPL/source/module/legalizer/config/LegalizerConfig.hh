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
 * @Date: 2023-02-09 09:36:07
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-09 10:01:21
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/config/LegalizerConfig.hh
 * @Description: LG config
 * 
 * 
 */
#ifndef IPL_LGCONFIG_H
#define IPL_LGCONFIG_H

#include <string>

namespace ipl {

class LGConfig{
public:
    LGConfig() = default;
    LGConfig(const LGConfig&) = default;
    LGConfig(LGConfig&&) = default;
    ~LGConfig() = default;

    LGConfig& operator=(const LGConfig&) = default;
    LGConfig& operator=(LGConfig&&) = default;

    // getter
    int32_t get_thread_num() const { return _thread_num;}
    int32_t get_global_padding() const { return _global_padding;}
    int32_t get_max_displacement() const { return _max_displacement;}

    // setter
    void set_thread_num(int32_t num_thread){ _thread_num = num_thread;}
    void set_global_padding(int32_t padding) { _global_padding = padding;}
    void set_max_displacement(int32_t max_displacement) { _max_displacement = max_displacement;}

private:
    int32_t _thread_num;
    int32_t _global_padding;
    int32_t _max_displacement;
};



}



#endif