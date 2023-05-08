/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-02 15:09:04
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-02 15:09:10
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/operation/LocalReorder.hh
 * @Description: Local reorder of detail placement
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
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