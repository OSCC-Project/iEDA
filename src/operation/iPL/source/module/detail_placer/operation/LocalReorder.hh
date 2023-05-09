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
 * @Date: 2023-03-02 15:09:04
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-02 15:09:10
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/operation/LocalReorder.hh
 * @Description: Local reorder of detail placement
 * 
 * 
 */
#ifndef IPL_LOCALREORDER_H
#define IPL_LOCALREORDER_H

#include <string>

#include "config/DetailPlacerConfig.hh"
#include "database/DPDatabase.hh"
#include "DPOperator.hh"

namespace ipl {
class LocalReorder
{
public:
    LocalReorder();
    LocalReorder(DPConfig* config, DPDatabase* database, DPOperator* dp_operator);
    LocalReorder(const LocalReorder&) = delete;
    LocalReorder(LocalReorder&&) = delete;
    ~LocalReorder();

    LocalReorder& operator=(const LocalReorder&) = delete;
    LocalReorder& operator=(LocalReorder&&) = delete;

    void runLocalReorder();

private:
    DPConfig* _config;
    DPDatabase* _database;
    DPOperator* _operator;
};
}
#endif