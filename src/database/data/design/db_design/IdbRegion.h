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
 * @file		IdbRegion.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Region information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"
#include "IdbInstance.h"

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class IdbRegion
{
 public:
  IdbRegion();
  ~IdbRegion();

  // getter
  const string& get_name() const { return _name; }
  const IdbRegionType& get_type() const { return _type; }
  std::vector<IdbRect*>& get_boundary() { return _boudary_list; }
  std::vector<IdbInstance*>& get_instance_list() { return _instance_list; }

  // setter
  void set_name(std::string name) { _name = name; }
  void set_type(IdbRegionType type) { _type = type; }
  void set_type(std::string type);
  IdbRect* add_boundary(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void clear_boundary();

  std::vector<IdbInstance*>& add_instance(IdbInstance* instance)
  {
    _instance_list.emplace_back(instance);
    return _instance_list;
  }

  // operator

 private:
  std::string _name;
  IdbRegionType _type;
  std::vector<IdbRect*> _boudary_list;
  //--------------tbd---------------
  // Property
  std::vector<IdbInstance*> _instance_list;
};

class IdbRegionList
{
 public:
  IdbRegionList();
  ~IdbRegionList();

  // getter
  const uint32_t get_num() { return _region_list.size(); }
  std::vector<IdbRegion*>& get_region_list() { return _region_list; }

  // setter
  IdbRegion* add_region(IdbRegion* region = nullptr);
  IdbRegion* add_region(std::string name);

  // operator
  IdbRegion* find_region(std::string name);

 private:
  std::vector<IdbRegion*> _region_list;
};

}  // namespace idb
