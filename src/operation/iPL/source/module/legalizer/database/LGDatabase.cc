/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-08 11:03:27
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:09:32
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGDatabase.cc
 * @Description:
 *
 *
 */
#include "LGDatabase.hh"

namespace ipl {

LGDatabase::LGDatabase()
{
}

LGDatabase::~LGDatabase()
{
  delete _lg_layout;

  for (auto pair : _lgCluster_map) {
    delete pair.second;
  }
  _lgCluster_map.clear();

  for (auto* inst : _lgInstance_list) {
    delete inst;
  }
  _lgInstance_list.clear();
  _lgInstance_map.clear();
  _instance_map.clear();
}

LGCluster* LGDatabase::findCluster(std::string name)
{
  LGCluster* cluster = nullptr;
  auto it = _lgCluster_map.find(name);
  if (it != _lgCluster_map.end()) {
    cluster = it->second;
  }

  return cluster;
}

void LGDatabase::insertCluster(std::string name, LGCluster* cluster)
{
  auto it = _lgCluster_map.find(name);
  if (it != _lgCluster_map.end()) {
    LOG_WARNING << "Cluster : " << name << " was added before";
  }
  _lgCluster_map.emplace(name, cluster);
}

void LGDatabase::deleteCluster(std::string name)
{
  auto it = _lgCluster_map.find(name);
  if (it != _lgCluster_map.end()) {
    _lgCluster_map.erase(it);
  } else {
    LOG_WARNING << "Cluster: " << name << " has not been insert";
  }
}

void LGDatabase::resetClusterInfo()
{
  // clean interval cluster info
  for (auto& interval_list : _lg_layout->get_interval_2d_list()) {
    for (auto* interval : interval_list) {
      interval->reset();
    }
  }

  // clean inst cluster info
  for (auto* inst : _lgInstance_list) {
    inst->set_belong_cluster(nullptr);
  }

  // delete cluster list
  for (auto pair : _lgCluster_map) {
    delete pair.second;
  }
  _lgCluster_map.clear();
}

}  // namespace ipl