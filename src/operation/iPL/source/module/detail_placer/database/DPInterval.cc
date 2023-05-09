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
#include "DPInterval.hh"

namespace ipl{

DPInterval::DPInterval(std::string name, int32_t min_x, int32_t max_x): _name(name), _belong_row(nullptr), _min_x(min_x), _max_x(max_x), _cluster_root(nullptr)
{
    _remain_length = max_x - min_x;
}

DPInterval::~DPInterval()
{

}

bool DPInterval::checkInLine(int32_t min_x, int32_t max_x){
    return (min_x >= _min_x && max_x <= _max_x);
}

void DPInterval::updateRemainLength(int32_t incr){
    _remain_length += incr;

    // Debug
    if(_remain_length < 0){
        LOG_WARNING << "remain_length is less than zero";
    }

    if(_remain_length > (_max_x - _min_x)){
        LOG_WARNING << "remain_length is more than origin";
    }
}

void DPInterval::resetInterval(){
    _cluster_root = nullptr;
    resetRemainLength();
}

void DPInterval::resetRemainLength(){
    _remain_length = _max_x - _min_x;
}

}