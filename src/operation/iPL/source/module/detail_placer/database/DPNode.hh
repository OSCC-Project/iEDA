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


#ifndef B72AD205_44D8_4879_8FE1_3CC283654C68
#define B72AD205_44D8_4879_8FE1_3CC283654C68

#include "DPInstance.hh"

namespace ipl{

class DPNode{

public:
    DPNode() = default;
    virtual ~DPNode() = default;

    void set_id(const int id) {_id = id;} 
    void set_inst(DPInstance* inst) {_inst = inst;}  

    int get_id() const { return _id;}
    DPInstance* get_inst() const {return _inst;}

    int64_t getPositionX() const {return _inst->get_coordi().get_x(); }
    int64_t getPositionY() const {return _inst->get_coordi().get_y(); }
    int64_t getInitialPositionX() const { return _inst->get_origin_shape().get_ll_x();}
    int64_t getInitialPositionY() const { return _inst->get_origin_shape().get_ll_y();}
    int64_t getWidth() const {return _inst->get_shape().get_width();}
    int64_t computeDisplacement() const { return _inst->computeDisplacement();}
    Rectangle<int64_t> getBound(){
        Rectangle<int32_t> bound32 = _inst->get_shape();
        Rectangle<int64_t> bound64 (
                static_cast<int64_t>(bound32.get_ll_x()),
                static_cast<int64_t>(bound32.get_ll_y()),
                static_cast<int64_t>(bound32.get_ur_x()),
                static_cast<int64_t>(bound32.get_ur_y())
        );
        return bound64;
    }

private:
    int _id = -1;
    DPInstance* _inst;
    DPNode* _left = nullptr;
    DPNode* _right = nullptr;
};

}



#endif /* B72AD205_44D8_4879_8FE1_3CC283654C68 */
