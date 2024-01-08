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

#ifndef IPL_POST_GP_DATABASE_H
#define IPL_POST_GP_DATABASE_H

#include "PlacerDB.hh"

namespace ipl {

class PostGPDatabase
{
 public:
  PostGPDatabase(): _placer_db(nullptr), _topo_manager(nullptr) {}
  PostGPDatabase(const PostGPDatabase&) = delete;
  PostGPDatabase(PostGPDatabase&&) = delete;
  ~PostGPDatabase() = default;

  PostGPDatabase& operator=(const PostGPDatabase&) = default;
  PostGPDatabase& operator=(PostGPDatabase&&) = default;

 private:
  PlacerDB* _placer_db;
  TopologyManager* _topo_manager;

  std::vector<Instance*> _inst_list;
  std::vector<Net*> _net_list;
  std::vector<Pin*> _pin_list;

  std::vector<Group*> _group_list;
  std::vector<NetWork*> _network_list;
  std::vector<Node*> _node_list;

  friend class PostGP;
};

}  // namespace ipl

#endif