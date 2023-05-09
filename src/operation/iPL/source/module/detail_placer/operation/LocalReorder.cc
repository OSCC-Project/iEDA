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
#include "LocalReorder.hh"

#include "module/logger/Log.hh"

namespace ipl{

LocalReorder::LocalReorder(DPConfig* config, DPDatabase* database, DPOperator* dp_operator)
{
    _config = config;
    _database = database;
    _operator = dp_operator;
}

LocalReorder::~LocalReorder()
{
}

void LocalReorder::runLocalReorder(){
    bool is_clusted = _operator->checkIfClustered();
    if(!is_clusted){
        _operator->updateInstClustering();
    }

    int64_t total_benefit = 0;
    auto& interval_2d_list = _database->get_layout()->get_interval_2d_list();

    // Debug
    int32_t row_index = 0;

    for(auto& interval_list : interval_2d_list){
        for(auto* interval : interval_list){

            auto* cur_cluster = interval->get_cluster_root();
            while(cur_cluster){
                auto inst_list = cur_cluster->get_inst_list();
                for(size_t i=0,j=i+1; i< inst_list.size() && j < inst_list.size(); i++,j++){
                    auto* inst_1 = inst_list[i];
                    auto* inst_2 = inst_list[j];
                    int64_t origin_hpwl = _operator->calInstPairAffectiveHPWL(inst_1, inst_2);

                    int32_t coordi_x = inst_1->get_coordi().get_x();
                    int32_t coordi_y = inst_1->get_coordi().get_y();

                    inst_2->updateCoordi(coordi_x, coordi_y);
                    inst_1->updateCoordi(coordi_x + inst_2->get_shape().get_width(), coordi_y);
                    int64_t modify_hpwl = _operator->calInstPairAffectiveHPWL(inst_1, inst_2);

                    if(origin_hpwl > modify_hpwl){
                        int32_t inst1_internal_id = inst_1->get_internal_id();
                        int32_t inst2_internal_id = inst_2->get_internal_id();
                        inst_1->set_internal_id(inst2_internal_id);
                        inst_2->set_internal_id(inst1_internal_id);
                        cur_cluster->replaceInstance(inst_2, inst1_internal_id);
                        cur_cluster->replaceInstance(inst_1, inst2_internal_id);
                        inst_list[i] = inst_2;
                        inst_list[j] = inst_1;
                        
                        total_benefit += (origin_hpwl - modify_hpwl);
                    }else{
                        inst_1->updateCoordi(coordi_x, coordi_y);
                        inst_2->updateCoordi(coordi_x + inst_1->get_shape().get_width(), coordi_y);
                    }
                }
                cur_cluster = cur_cluster->get_back_cluster();
            }

        }
        row_index += 1;
    }

    // LOG_INFO << "Expected HPWL Benefit: " << total_benefit;
}


}