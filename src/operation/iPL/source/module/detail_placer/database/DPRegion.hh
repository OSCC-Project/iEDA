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
 * @Date: 2023-03-03 19:18:00
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-06 11:25:58
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPRegion.hh
 * @Description: Region info of detail placement
 * 
 * 
 */
#ifndef IPL_DPREGION_H
#define IPL_DPREGION_H

#include <string>
#include <vector>

#include "data/Rectangle.hh"

#include "DPInstance.hh"

namespace ipl {

enum class DPREGION_TYPE{
    kNone,
    kFence,
    kGuide
};

class DPRegion
{
public:
    DPRegion() = delete;
    explicit DPRegion(std::string name);
    DPRegion(const DPRegion&) = delete;
    DPRegion(DPRegion&&) = delete;
    ~DPRegion();

    DPRegion& operator=(const DPRegion&) = delete;
    DPRegion& operator=(DPRegion&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    DPREGION_TYPE get_type() const { return _type;}
    const std::vector<Rectangle<int32_t>>& get_shape_list() const { return _shape_list;}
    const std::vector<DPInstance*>& get_inst_list() const { return _inst_list;}

    // setter
    void set_type(DPREGION_TYPE type) { _type = type;}
    void add_shape(Rectangle<int32_t> shape) { _shape_list.push_back(shape);}
    void add_inst(DPInstance* inst) { _inst_list.push_back(inst);}


private:
    std::string _name;
    DPREGION_TYPE _type;
    std::vector<Rectangle<int32_t>> _shape_list;
    std::vector<DPInstance*> _inst_list;

};
}
#endif