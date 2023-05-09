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
 * @Date: 2023-02-08 11:03:27
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:09:44
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGDatabase.hh
 * @Description: LG database
 * 
 * 
 */
#ifndef IPL_LGDATABASE_H
#define IPL_LGDATABASE_H

#include <string>
#include <vector>
#include <map>

#include "PlacerDB.hh"
#include "LGInstance.hh"
#include "LGCluster.hh"
#include "LGLayout.hh"

namespace ipl {
class LGDatabase
{
public:
    LGDatabase();
    LGDatabase(const LGDatabase&) = delete;
    LGDatabase(LGDatabase&&) = delete;
    ~LGDatabase();

    LGDatabase& operator=(const LGDatabase&) = delete;
    LGDatabase& operator=(LGDatabase&&) = delete;

    LGCluster* findCluster(std::string name);
    void insertCluster(std::string name, LGCluster* cluster);
    void deleteCluster(std::string name);

    void resetClusterInfo();
private:
    PlacerDB* _placer_db;

    int32_t _shift_x;
    int32_t _shift_y;

    LGLayout* _lg_layout;
    std::vector<LGInstance*> _lgInstance_list;
    std::map<std::string, LGCluster*> _lgCluster_map;
    std::map<LGInstance*, Instance*> _lgInstance_map;
    std::map<Instance*, LGInstance*> _instance_map;

    friend class AbacusLegalizer;
};
}
#endif