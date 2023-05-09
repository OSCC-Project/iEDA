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
 * @Date: 2023-03-02 15:18:25
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-02 15:24:19
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/operation/BinOpt.hh
 * @Description: Bin Opt of detail placement
 * 
 * 
 */
#ifndef IPL_BINOPT_H
#define IPL_BINOPT_H

#include <string>
#include <map>

#include "config/DetailPlacerConfig.hh"
#include "database/DPDatabase.hh"
#include "DPOperator.hh"

namespace ipl {
class BinOpt
{
public:
    BinOpt();
    BinOpt(DPConfig* config, DPDatabase* database, DPOperator* dp_operator);
    BinOpt(const BinOpt&) = delete;
    BinOpt(BinOpt&&) = delete;
    ~BinOpt();

    BinOpt& operator=(const BinOpt&) = delete;
    BinOpt& operator=(BinOpt&&) = delete;

    void runBinOpt();

private:
    DPConfig* _config;
    DPDatabase* _database;
    DPOperator* _operator;
    int32_t _row_height;
    int32_t _site_width;

    void slidingInstBetweenGrids(Grid* supply_grid, Grid* demand_grid, int64_t grid_area);
    int64_t calSlidingFlowValue(Grid* supply_grid, Grid* demand_grid, int64_t grid_area);
    DPCluster* obtainIntervalFirstCluster(DPInterval* interval);
    DPCluster* obtainIntervalLastCluster(DPInterval* interval);
    bool moveInstToInterval(DPInstance* inst, DPInterval* interval);

    void instantLegalizeCluster(DPCluster* cluster);
    void arrangeClusterMinXCoordi(DPCluster* cluster);
    int32_t obtainFrontMaxX(DPCluster* cluster);
    int32_t obtainBackMinX(DPCluster* cluster);
    void collapseCluster(DPCluster* dest_cluster, DPCluster* src_cluster);
    DPCluster* createInstClusterForInterval(DPInstance* inst, DPInterval* interval);

};
}
#endif