/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-08 11:03:27
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:09:44
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGDatabase.hh
 * @Description: LG database
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
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