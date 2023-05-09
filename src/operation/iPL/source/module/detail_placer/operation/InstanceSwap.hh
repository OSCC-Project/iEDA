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
 * @Date: 2023-03-02 14:56:19
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-02 14:59:36
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/operation/InstanceSwap.hh
 * @Description: Swap of detail placement
 * 
 * 
 */
#ifndef IPL_INSTANCESWAP_H
#define IPL_INSTANCESWAP_H

#include <string>

#include "config/DetailPlacerConfig.hh"
#include "database/DPDatabase.hh"
#include "DPOperator.hh"

namespace ipl {
class InstanceSwap
{
public:
    InstanceSwap();
    InstanceSwap(DPConfig* config, DPDatabase* database, DPOperator* dp_operator);
    InstanceSwap(const InstanceSwap&) = delete;
    InstanceSwap(InstanceSwap&&) = delete;
    ~InstanceSwap();

    InstanceSwap& operator=(const InstanceSwap&) = delete;
    InstanceSwap& operator=(InstanceSwap&&) = delete;

    void runGlobalSwap();
    void runVerticalSwap();

private:
    DPConfig* _config;
    DPDatabase* _database;
    DPOperator* _operator;
    int32_t _row_height;
    int32_t _site_width;

    void sortInstBasedHPWLBenefit(std::vector<DPInstance*>& movable_inst_list);
    void searchCandidateCoordiList(Rectangle<int32_t>& optimal_region, DPInstance* inst, std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list);
    void searchImproveYCoordiList(std::pair<int32_t, int32_t>& optimal_line, DPInstance* inst, int32_t row_range, std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list);
    void fillIntervalCandidateList(DPInterval* interval, int32_t query_min, int32_t query_max, int32_t inst_width, std::vector<std::pair<Point<int32_t>, DPInstance*>>& candidate_list);

    int64_t placeInstance(DPInstance* inst, int32_t x_coordi, int32_t y_coordi, bool is_trial);
    int64_t swapInstance(DPInstance* inst_1, DPInstance* inst_2, bool is_trial);
    
    void updateAloneInstToInterval(DPInstance* inst, DPInterval* interval);
    void instantLegalizeCluster(DPCluster& cluster);    
    void arrangeClusterMinXCoordi(DPCluster& cluster_ref);
    int32_t obtainFrontMaxX(DPCluster& cluster);
    int32_t obtainBackMinX(DPCluster& cluster);
    
    void updateClusterInstCoordi(DPCluster& cluster, std::vector<DPInstance*>& special_insts, bool is_trial);
    int64_t calOtherInstMovement(DPCluster& cluster, std::vector<DPInstance*>& except_insts);
    void replaceCluster(DPCluster& origin_cluster, DPCluster& modify_cluster);
    void replaceClusterPair(DPCluster& dest_cluster_1, DPCluster& src_cluster_1,DPCluster& dest_cluster_2, DPCluster& src_cluster_2);

    DPInterval* obtainCurrentInterval(DPInstance* inst);
    DPInterval* obtainCorrespondingInterval(Rectangle<int32_t>& inst_shape);
    void temporarySpliceCluster(DPCluster& dest_cluster, DPCluster& src_cluster);

    bool checkIfTwoClusterFusion1(DPCluster& cluster, DPInstance* inst_1, DPInstance* inst_2);
    bool checkIfTwoClusterFusion2(DPCluster& cluster_1, DPCluster& cluster_2);

    void eraseInstAndSplitCluster(DPCluster* cluster, DPInstance* inst);

    // tmp test
    int64_t testCalTotalHPWL();

};
}
#endif