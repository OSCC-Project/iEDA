/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-07 21:18:56
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 14:56:02
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGInterval.cc
 * @Description: 
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
 */
#include "LGInterval.hh"

namespace ipl{

LGInterval::LGInterval(std::string name, int32_t min_x, int32_t max_x): _name(name), _cluster_root(nullptr), _belong_row(nullptr), _min_x(min_x), _max_x(max_x), _remain_length(max_x - min_x){}

LGInterval::~LGInterval()
{

}

void LGInterval::updateRemainLength(int32_t occupied_length){
    _remain_length += occupied_length;
}

void LGInterval::reset(){
    _cluster_root = nullptr;
    _remain_length = (_max_x - _min_x);
}


}