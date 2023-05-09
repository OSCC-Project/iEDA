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
#pragma once
/**
 * @project		iDB
 * @file		IdbGroup.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Defines groups in a design.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "IdbInstance.h"

namespace idb {

class IdbRegion;

class IdbGroup
{
 public:
  IdbGroup(std::string group_name);
  ~IdbGroup();

  // getter
  const std::string& get_group_name() const { return _group_name; }
  IdbRegion* get_region() { return _region; }
  IdbInstanceList* get_instance_list() { return _instance_list; }

  // setter
  void set_group_name(std::string name) { _group_name = name; }
  void set_region(IdbRegion* region) { _region = region; }
  void add_instance(IdbInstance* instance) { _instance_list->add_instance(instance); }

 private:
  std::string _group_name;
  IdbRegion* _region;
  IdbInstanceList* _instance_list;
};

class IdbGroupList
{
 public:
  IdbGroupList();
  ~IdbGroupList();

  // getter
  const int32_t get_num() const { return _group_list.size(); };
  std::vector<IdbGroup*>& get_group_list() { return _group_list; }

  // setter
  IdbGroup* add_group(std::string name);

  void reset();

  // operator

 private:
  int32_t _num;
  std::vector<IdbGroup*> _group_list;
};

}  // namespace idb
